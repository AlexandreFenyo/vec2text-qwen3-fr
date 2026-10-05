"""Quiet, offline-first environment for the CLI tools. Import before transformers / sentence_transformers."""
import os

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
REQUIRED = ["models--Qwen--Qwen3-Embedding-0.6B", "models--Qwen--Qwen3-0.6B-Base"]


def quiet_offline():
    """No Hub round-trips when the models are already cached, no progress bars, no tokenizer-parallelism warning."""
    hub = os.path.join(HF_CACHE, "hub")
    if all(os.path.isdir(os.path.join(hub, r)) for r in REQUIRED):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import transformers
        transformers.utils.logging.disable_progress_bar()
        transformers.utils.logging.set_verbosity_error()
    except Exception:
        pass
