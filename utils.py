import logging
import sys
from textwrap import TextWrapper

import datasets
import huggingface_hub
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import torch
import transformers

try:
    from IPython.display import set_matplotlib_formats
except ImportError:
    def set_matplotlib_formats(*args, **kwargs):
        pass

is_colab = "google.colab" in sys.modules
is_gpu_available = torch.cuda.is_available()


def install_mpl_fonts():
    font_dir = ["./orm_fonts/"]
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)


def set_plot_style():
    install_mpl_fonts()
    set_matplotlib_formats("pdf", "svg")
    plt.style.use("plotting.mplstyle")
    logging.getLogger("matplotlib").setLevel(level=logging.ERROR)


def display_library_version(library):
    print(f"Using {library.__name__} v{library.__version__}")


def setup_chapter():
    if not is_gpu_available:
        print("GPUが検出されませんでした！このノートブックはGPUなしでは*非常に*遅くなる可能性があります 🐢")
        if is_colab:
            print("ランタイム > ランタイムのタイプを変更 でGPUハードウェアアクセラレータを選択してください。")
    display_library_version(transformers)
    display_library_version(datasets)
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    try:
        huggingface_hub.logging.set_verbosity_error()
    except:
        pass
    set_plot_style()


def wrap_print_text(print):
    """Adapted from: https://stackoverflow.com/questions/27621655/how-to-overload-print-function-to-expand-its-functionality/27621927"""

    def wrapped_func(text):
        if not isinstance(text, str):
            text = str(text)
        wrapper = TextWrapper(
            width=80,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        return print("\n".join(wrapper.fill(line) for line in text.split("\n")))

    return wrapped_func


print = wrap_print_text(print)
