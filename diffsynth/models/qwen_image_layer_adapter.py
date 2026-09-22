import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def attention(
    q,
    k,
    v,
    mode="torch",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    max_seqlen_q=None,
    batch_size=1,
):
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")



class LayerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.scale = qk_scale or head_dim**-0.5
       
        self.linear1_kv = nn.Linear(hidden_size, hidden_size * 2, **factory_kwargs)
        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        self.linear2 = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm_feat = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.pre_norm_motion = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        layer_split_vec: torch.Tensor,
        layer_split_mask: Optional[torch.Tensor] = None,
        use_context_parallel=False,
    ) -> torch.Tensor:
        
        """
        B (N H W) (C P Q)->B L D
        
        out:B (N H W) (C P Q)
        """
        
        B,L,D = layer_split_vec.shape

        x_motion = self.pre_norm_motion(layer_split_vec)
        x_feat = self.pre_norm_feat(x)

        kv = self.linear1_kv(x_motion)
        q = self.linear1_q(x_feat)

        k, v = rearrange(kv, "B L (K H D) -> K B L H D", K=2, H=self.heads_num)
        q = rearrange(q, "B S (H D) -> B S H D", H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        k = rearrange(k, "B L H D -> B H L D")  
        v = rearrange(v, "B L H D -> B H L D") 

        q = rearrange(q, "B L H D -> B H L D")  
        # Compute attention.
        attn = F.scaled_dot_product_attention(q, k, v)

        attn = rearrange(attn, "B H S D -> B S (H D)")

        output = self.linear2(attn)

        if layer_split_mask is not None:
            output = output * rearrange(layer_split_mask, "B T H W -> B (T H W)").unsqueeze(-1)

        return output

# def custom_qr(input_tensor):
#     original_dtype = input_tensor.dtype
#     if original_dtype == torch.bfloat16:
#         q, r = torch.linalg.qr(input_tensor.to(torch.float32))
#         return q.to(original_dtype), r.to(original_dtype)
#     return torch.linalg.qr(input_tensor)


class LayerAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        num_adapter_layers: int = 1,
        dtype=None,
        device=None,
    ):

        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.hidden_size = hidden_dim
        self.heads_num = heads_num
        
        self.img_in = nn.Linear(64, 3072)
        self.fuser_blocks = nn.ModuleList(
            [
                LayerBlock(
                    self.hidden_size,
                    self.heads_num,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(num_adapter_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_split_vec: torch.Tensor,
        idx: int,
        freqs_cis_q: Tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_k: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.fuser_blocks[idx](x, layer_split_vec, freqs_cis_q, freqs_cis_k)


class QwenImageLayerAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_adapter = LayerAdapter(heads_num=24, hidden_dim=3072, num_adapter_layers=60 // 5)
    def after_patch_embedding(self, layer_split_vec):
        return self.layer_adapter.img_in(layer_split_vec)
    
    def after_transformer_block(self, block_idx, x, layer_split_vec, layer_split_mask=None):
        if block_idx % 5 == 0:
            #rearrange
            adapter_args = [x, layer_split_vec, layer_split_mask,False]
            residual_out = self.layer_adapter.fuser_blocks[block_idx // 5](*adapter_args)
            x = residual_out + x
        return x


