按照agent.md要求生成或修改文件
Generate or modify files according to agent.md requirements.

examples/layer_edit/model_training/train.py"DiT 的不同块分配不同损失训练：整体上的flow_match损失不变，额外约束：1层（RGB loss 0.2,A loss 0.8）+2层(RGB 0.8,A 0.2),能在一轮训练中实现"第 1 个块（RGB 0.2 / A 0.8）+ 第 2/3 个块（RGB 0.8 / A 0.2）"的差异化训练，循环设计3*10=30层DIT类似于A,RGB,RGB,A,RGB,RGB这样循环,在这3层中，1层有self,text attention,mask attention,2/3层只有self attention"将这个作为损失的默认设置,
"""
examples/layer_edit/model_training/train.py
Differentiated loss training for different DiT blocks: global flow‑match loss stays unchanged.
Extra constraints: block‑1 uses RGB loss weight 0.2, Alpha loss weight 0.8;
blocks‑2/3 use RGB loss weight 0.8, Alpha loss weight 0.2.
Supports differentiated training in one training iteration.
Use cyclic pattern A,RGB,RGB repeated for the 3*10=30‑layer DiT.
Within each triplet:
- 1st block: self‑attention + text‑cross‑attention + mask‑attention
- 2nd & 3rd blocks: self‑attention only
Set this cyclic pattern as the default loss configuration.
"""

models/train/LayerEdit_full/epoch-0.safetensors需要用推理代码examples/layer_edit/model_inference_low_vram/.py验证，data/combine/test.json中的提示词和layer_input_image可以使用