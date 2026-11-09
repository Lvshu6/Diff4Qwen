停止该运行的所有进程：
pkill -TERM -f "[l]ayer_edit/model_training/train.py"
检查是否还有残留：
pgrep -af "[l]ayer_edit/model_training/train.py"
