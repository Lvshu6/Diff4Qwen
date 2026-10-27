创建新模型架构：LayerEditDit,用于图->Live2D图层psd，分割生成任务
diffsynth/models/wan_video_dit.py :借鉴wan_video的Fused overwrite的Dit设计,推理时，输入:layer_input_image,像图生视频一样后面拼接待生成噪声,将视频生成移植为多图层生成,时间维度变成图层顺序,1+n个(1是layer_input_image (overwrite),n是分割出的多图层),生成新的,去掉多余vace,controlnet,audio,animate,wan2dance等插件unit的功能干净完整的脚本diffsynth/models/layer_edit_dit.py,此外注意patchify采用无参数，2×2 patch 摊平到通道的张量重排,设计dit.img_in转换维度到dit

diffsynth/models/qwen_image_vae.py:参考rgba vae ,image_channel为4,我将直接使用qwen_image_layer 的rgba vae

diffsynth/models/qwen_image_text_encoder.py:我将直接使用qwen_image_layer的text_encoder

我要预训练一个模型，创建模型权重初始化脚本dit_init.py,bf16,.safetensors
examples/wanvideo/model_training/train.py:参考生成examples/layer_edit/model_training/train.py
examples/qwen_image/model_training/Qwen-Image-Layered_distill.sh参考生成examples/layer_edit/model_training/layer_edit_full.sh
沿用examples/wanvideo/model_training/full/accelerate_config_14B.yaml
examples/wanvideo/model_training/full/accelerate_config_zero3.yaml到layer_edit
模型细节：
Unit设计:"mask"键:list[Image:RGB/RGBA]对齐1+n,缺失默认全白->只取RGB值,白色的为1,黑色的为0，按RGB亮度转换为list[1通道的值范围(0,1)]->down_to_latent_size,1通道复制为4通道->在维度上latent (patchify之前)concat mask->patchify->x
参考diffsynth/pipelines/qwen_image.py处理layerinputimage用于单独vae编码作为额外条件,它的mask默认全1以对齐通道
参考examples/wanvideo/model_training/train.py设计训练
我希望加一个约束：vae解码出训练预测的RGBA图像，计算RGB损失和A损失，在指定的DIT层中约束，我只希望RGB损失和A损失约束影响指定层不影响其他层，不是输出dit中间层，是输出dit最终结果后vae解码算loss然后只约束特定dit层。
目标：
1. DiT 完整前向传播得到最终 latent 输出；
2. VAE 解码 DiT最终输出 latent得到 RGBA 预测图像；
3. 把预测 RGBA 和 GT RGBA 分别拆开，计算 RGB loss + Alpha (A) loss；
4. 该 loss 只回传更新指定的若干 DiT block 层参数，其余 DiT 层不参与这个 RGBA/A loss 的梯度更新；
DiT 的不同块分配不同损失训练：整体上的flow_match损失不变，额外约束：1层（RGB loss 0.2,A loss 0.8）+2层(RGB 0.8,A 0.2),能在一轮训练中实现"第 1 个块（RGB 0.2 / A 0.8）+ 第 2/3 个块（RGB 0.8 / A 0.2）"的差异化训练，循环设计3*10=30层DIT类似于A,RGB,RGB,A,RGB,RGB这样循环,在这3层中，1层有self,text attention,2/3层只有self attention
分两种损失训练模式普遍的examples/wanvideo/model_training/train.py的flow_match和先前我说的,sh可以用一个参数控制

layerinputimage vae后的first_frame_latent还可以同样加上全1掩码通道，rerange,和rerange后的latent在维度上concat增加了维度后再通过img_in的映射.