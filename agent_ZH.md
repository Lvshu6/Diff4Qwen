创建新模型架构：LayerEditDit,用于图->Live2D图层psd，分割生成任务
layer_edit_dit 修改成只删掉qwenimagelayer整套文本流，文本 token 向量文本经过text_embedding之后从此不再被任何网络层更新,权重初始化也从qwenimagelayer权重获得。推理时，输入:layer_input_image,像图生视频一样后面拼接待生成噪声,将视频生成移植为多图层生成,时间维度变成图层顺序,1+n个(1是layer_input_image (overwrite),n是分割出的多图层)

diffsynth/models/qwen_image_vae.py:参考rgba vae ,image_channel为4,我将直接使用qwen_image_layer 的rgba vae

diffsynth/models/qwen_image_text_encoder.py:我将直接使用qwen_image_layer的text_encoder

创建模型权重初始化脚本dit_init.py,bf16,.safetensors从qwenimagelayer权重获得
examples/layer_edit/model_training/train.py
examples/qwen_image/model_training/Qwen-Image-Layered_distill.sh参考生成examples/layer_edit/model_training/layer_edit_full.sh
沿用examples/qwen/model_training/full/accelerate_config_14B.yaml
examples/qwen/model_training/full/accelerate_config_zero3.yaml到layer_edit
模型细节：
Unit设计:"mask"键:list[Image:RGB/RGBA]对齐1+n,缺失默认全白->只取RGB值,白色的为1,黑色的为0，按RGB亮度转换为list[1通道的值范围(0,1)]->down_to_latent_size,1通道复制为4通道->latent (patchify之前)concat mask->patchify->x
参考diffsynth/pipelines/qwen_image.py处理layerinputimage用于单独vae编码作为额外条件,它的mask默认全1以对齐通道
layerinputimage vae后的first_frame_latent还可以同样加上全1掩码通道，rerange,和rerange后的latent在序列 token 维度拼接再通过img_in的映射.mask设计与qwenimagelayer不同，权重初始化可以复制后再没有的部分随机初始化

