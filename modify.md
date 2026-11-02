我想写一个脚本为了layer_edit的初始化,复用Qwen_image_layer的权重,但他们的权重是不一样的，我需要在不改变他们的模型基本架构的同时复用一些权重，可以尝试平均复制之类的方法,可修改layer_edit_dit的dim和qwenimagelayer一致方便复制，但不可将它的基本设计思想修改，层数定为48层(3*16)

git add .
git commit -m "Add initial dataset"
git push

I want to write an initialization script for layer_edit that reuses weights from Qwen_image_layer. The two models have different weight structures, so I need to reuse compatible weights without changing their basic model architecture. I’m open to strategies like copying, averaging, tiling, or zero-padding the weights where the shapes don’t match. I can also adjust the dim of layer_edit_dit to match Qwen_image_layer to make weight copying easier, but I don’t want to change its core design. The number of layers should be fixed at 48 (3 × 16).


我希望layer_edit_dit 修改成只删掉qwenimagelayer整套文本流，文本 token 向量文本经过text_embedding之后从此不再被任何网络层更新,权重初始化也变得更简单了,恢复之前Qwen 为每一个模块都配备独立的时间步投影层