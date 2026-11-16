# Implementation of Time‑Weighted Regularization for Alpha Loss

1. Compute the alpha loss in pixel space for **all timesteps**.
2. Weight \(w(t)\): assign higher weights in the early denoising phase, with weights decaying smoothly toward zero in the late denoising phase.

Implement this strategy rationally with consideration of practical numerical magnitudes, and adopt reasonable hyperparameter settings.