import subprocess
import sys

is_colab = "google.colab" in sys.modules
is_kaggle = "kaggle_secrets" in sys.modules
# torch-scatter binaries depend on the torch and CUDA version, so we define the
# mappings here for Colab & Kaggle
torch_to_cuda = {"1.10.0": "cu113", "1.9.0": "cu111", "1.9.1": "cu111"}


def install_requirements(
    is_chapter2: bool = False, 
    is_chapter6: bool = False,
    is_chapter7: bool = False,
    is_chapter7_v2: bool = False,
    is_chapter10: bool = False,
    is_chapter11: bool = False
    ):
    """プロジェクトに必要なパッケージをインストールします。"""

    print("⏳ 基本要件をインストール中...")
    cmd = ["python", "-m", "pip", "install", "-r"]
    if is_chapter7:
        cmd += "requirements-chapter7.txt -f https://download.pytorch.org/whl/torch_stable.html".split()
    elif is_chapter7_v2:
        cmd.append("requirements-chapter7-v2.txt")
    else:
        cmd.append("requirements.txt")
    process_install = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_install.returncode != 0:
        raise Exception("😭 基本要件のインストールに失敗しました")
    else:
        print("✅ 基本要件がインストールされました！")
    print("⏳ Git LFSをインストール中...")
    process_lfs = subprocess.run(["apt", "install", "git-lfs"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process_lfs.returncode == -1:
        raise Exception("😭 Git LFSとsoundfileのインストールに失敗しました")
    else:
        print("✅ Git LFSがインストールされました！")

    if is_chapter2:
        transformers_cmd = "python -m pip install transformers>=4.21.0 datasets>=2.0.0".split()
        process_scatter = subprocess.run(
            transformers_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    if is_chapter6:
        transformers_cmd = "python -m pip install datasets>=2.0.0".split()
        process_scatter = subprocess.run(
            transformers_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    if is_chapter10:
        wandb_cmd = "python -m pip install wandb".split()
        process_scatter = subprocess.run(
            wandb_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    if is_chapter11:
        import torch

        torch_version = torch.__version__.split("+")[0]
        print(f"⏳ torch v{torch_version}用のtorch-scatterをインストール中...")
        if is_colab:
            torch_scatter_cmd = f"python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch_version}+{torch_to_cuda[torch_version]}.html".split()
        else:
            torch_scatter_cmd = "python -m pip install torch-scatter".split()
        process_scatter = subprocess.run(
            torch_scatter_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process_scatter.returncode == -1:
            raise Exception("😭 torch-scatterのインストールに失敗しました")
        else:
            print("torch-scatterがインストールされました！")
        print("⏳ soundfileをインストール中...")
        process_audio = subprocess.run(
            ["apt", "install", "libsndfile1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if process_audio.returncode == -1:
            raise Exception("😭 soundfileのインストールに失敗しました")
        else:
            print("✅ soundfileがインストールされました！")
        print("🥳 章のインストールが完了しました！")
