import subprocess

def nvidia_reset_gpu():
    # 1. 打印状态
    out = subprocess.check_output(["nvidia-smi"], text=True)
    print(out)

    # 2. 杀掉所有持有cuda上下文的用户进程（兜底）
    subprocess.run("fuser -v /dev/nvidia* 2>/dev/null | awk '{print $2}' | sort -u | xargs -I{} kill -9 {} 2>/dev/null", shell=True)

    # 3. 重置GPU，清除僵尸CUDA上下文，root才有权限
    print("\n=== Performing GPU reset ===")
    subprocess.run(["nvidia-smi", "-r"], check=False)

    # 再次查看显存
    print("\n=== After reset ===")
    print(subprocess.check_output(["nvidia-smi"], text=True))


if __name__ == "__main__":
    nvidia_reset_gpu()