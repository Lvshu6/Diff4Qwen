# Alpha 损失时间加权正则实现（Rectified Flow，Qwen‑Image‑Layered）

需求要点：

1. **全部时间步都计算像素空间的 alpha 损失，不做硬截断**；
2. 权重 \(w(t)\)：**t 越小权重越高，t 增大权重平滑衰减，永远大于 0**；
3. 总损失 = 流匹配 MSE 损失 + \(w(t) \times \mathcal{L}_\mathrm{alpha}\)；
4. \(t\in[0,1]\)，t=0 为目标真实样本，t=1 为纯噪声。