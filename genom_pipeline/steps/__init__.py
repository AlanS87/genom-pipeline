# Import every third-party dependency the six step modules below need --
# plus textdistance, which fusion.py pulls in transitively via
# genom_pipeline.exact_matcher/bertmaplt_string.py -- *before* importing
# `.extract` starts DeepOnto's JVM.
#
# `.extract` starts the JVM as a side effect of being imported (see
# genom_pipeline/_deeponto_compat.py). Once the JVM is up, JPype installs a
# global sys.meta_path import hook that is consulted for *every* import
# anywhere in the process from then on, not just Java ones. JPype's own
# docs warn that resolving a heavy/native extension's submodules through
# that hook for the *first* time can segfault outright instead of raising a
# catchable exception. Because this file eagerly imports all six step
# modules one after another (extract, define, embed, retrieve, judge,
# fusion -- and fusion.py in turn imports genom_pipeline.exact_matcher,
# which pulls in bertmaplt_string.py's textdistance too), anyone importing
# genom_pipeline.steps -- directly, via `from genom_pipeline.steps.X import
# run`, or transitively via genom_pipeline.pipeline.run_pipeline() -- was at
# risk of hitting exactly that: extract.py starts the JVM, and then
# define.py's transformers, embed.py's tiktoken/openai, retrieve.py's faiss
# and judge.py's transformers were each being imported for the first time
# *after* that, in the same breath. Importing them all here first means
# none of them is ever "the first import after the JVM starts".
import numpy  # noqa: F401
import pandas  # noqa: F401
import torch  # noqa: F401
import transformers  # noqa: F401
import tqdm  # noqa: F401
import textdistance  # noqa: F401

try:
    import tiktoken  # noqa: F401
except Exception:
    pass

try:
    import faiss  # noqa: F401
except Exception:
    pass

try:
    import openai  # noqa: F401
except Exception:
    pass

from .extract import run as extract
from .define import run as define
from .embed import run as embed
from .retrieve import run as retrieve
from .judge import run as judge
from .fusion import run as fuse

__all__ = ["extract", "define", "embed", "retrieve", "judge", "fuse"]
