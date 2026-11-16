# GPU 显存占用排查与释放

## 本次排查结果

检查时，RTX 6000D 显存占用约为 73,860 MiB，GPU 利用率为 0%。占用来自 `examples/layer_edit/model_training/train.py` 训练任务遗留的 6 个 Python 数据加载工作进程，而不是 Xorg（Xorg 仅占用约 4 MiB）。训练启动进程已退出，但子进程仍存活并持有 `/dev/nvidia0`、`/dev/nvidia-uvm`，所以显存没有释放；`nvidia-smi` 进程表还显示了已经不存在的 PID，这是进程退出后残留状态与仍存活子进程不一致的表现。

已向这 6 个工作进程发送 TERM，工作进程随后退出。但复查时 GPU 仍报告 73,860 MiB，`nvidia-smi` 中的计算进程 PID 已不存在，`fuser -v /dev/nvidia*` 也只显示 Xorg。也就是说，训练进程已停止，但驱动仍报告显存占用；剩余的僵尸进程不持有设备文件。因为这是共享 GPU，未尝试重置显卡，以免中断其他用户。若此状态持续，需由管理员确认无其他作业后再排查或重置 GPU/驱动。

## 以后遇到显存占用时

1. 先查看显存和 GPU 进程：

   ```bash
   nvidia-smi
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
   ```

   GPU 利用率低不代表显存没有被进程占用。进程名显示 `[Not Found]` 时，也不要只依据它判断进程已经结束。

2. 查找实际打开 NVIDIA 设备的进程，并核对命令、用户和启动时间：

   ```bash
   sudo fuser -v /dev/nvidia*
   ps -o pid,ppid,pgid,user,stat,etime,cmd -p <PID>
   ```

   如果 `nvidia-smi` 没有列出完整进程，`fuser` 通常仍能找到持有设备文件的进程。确认进程身份和归属后再操作；共享机器上不要终止其他用户或系统进程。

3. 优先回到启动训练的终端按 `Ctrl-C` 正常停止。若启动终端已不存在，向确认无误的进程发送 TERM：

   ```bash
   kill -TERM <PID>
   ```

   对确认属于同一个遗留任务的进程组，可以先查看组内成员：

   ```bash
   ps -eo pid,ppid,pgid,user,stat,cmd | awk '$3 == <PGID>'
   ```

   确认该组只包含目标任务后，再优雅停止整个进程组：

   ```bash
   kill -TERM -- -<PGID>
   ```

4. 等待几秒后重新检查：

   ```bash
   nvidia-smi
   sudo fuser -v /dev/nvidia*
   ```

   只有目标进程仍存在且确认可以强制结束时，才对具体 PID 使用 `kill -KILL <PID>`，然后再次验证显存是否释放。若所有持有设备的计算进程都已退出但显存仍显示被占用，不要反复杀进程；让管理员检查驱动状态，并在确认不会影响其他作业后处理。

## 避免再次发生

- 训练结束时正常停止启动进程，并确认 DataLoader 等子进程也退出。
- 使用 `tmux`、`screen` 或作业调度器运行长任务，保留会话/作业 ID，避免失去对后台进程的管理。
- 不要使用 `killall python` 或按名称批量杀进程；这可能误杀其他任务。
- 显存占用仍在但进程已全部退出时，先重新检查设备持有者；不要直接重启驱动或机器。在共享服务器上联系管理员处理驱动层面的残留状态。
