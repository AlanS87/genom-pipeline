"""
pytest configuration for the genom-pipeline test suite.

Import-order fix for a real segfault seen on CSF3. The first version of this
file only imported torch, which fixed the *specific* crash reported then
(torch.cuda being resolved by JPype's import hook for the first time), but
the same class of crash re-appeared one step later with a different
package (transformers -> httpx -> mimetypes) -- see below for why fixing it
one import at a time doesn't converge, and why this file now pre-imports
the *whole* set of third-party dependencies the pipeline uses.

Root cause: once genom_pipeline._deeponto_compat starts the JVM
(jpype.isJVMStarted() -> deeponto.init_jvm()), JPype installs a global
sys.meta_path import hook that gets consulted for *every* import anywhere
in the process from then on, not just Java ones. JPype's own docs warn that
if a heavy/native extension is imported for the *first* time after that
point, resolving one of its submodules through that hook can segfault
outright rather than raise a catchable exception. Two different instances
of exactly this were observed on CSF3:

    File ".../jpype/imports.py", line 174, in find_spec
    File ".../torch/cuda/__init__.py", line 68, in <module>
    File ".../torch/__init__.py", line 2585, in <module>
    File "tests/test_define_batching.py", line 25, in <module>
    Fatal Python error: Segmentation fault

and, after the torch-only fix below was added:

    File ".../jpype/imports.py", line 174, in find_spec
    File ".../mimetypes.py", line 32, in <module>
    File ".../httpx/_multipart.py", line 4, in <module>
    ... (httpx -> huggingface_hub -> transformers) ...
    Fatal Python error: Segmentation fault

The actual trigger in both cases is genom_pipeline/steps/__init__.py's own
import order: it does `from .extract import run as extract` first (which
starts the JVM as a side effect), then immediately continues on to
`from .define import run as define`, `from .embed import run as embed`,
etc. in the same breath -- and those import transformers, tiktoken, openai,
faiss and pandas for the first time in the process, right after the JVM has
already started. Any test (or real user) that imports
genom_pipeline.steps.* -- directly or via genom_pipeline.pipeline.
run_pipeline() -- was exposed to this, which is why
genom_pipeline/steps/__init__.py has now been reordered to import all of
these third-party packages *before* `.extract` ever runs (see that file).

That fix alone covers every path that goes through genom_pipeline.steps.
The one path that doesn't is tests/test_deeponto_compat.py, which imports
genom_pipeline._deeponto_compat *directly* (to test it in isolation) and so
starts the JVM without ever going through genom_pipeline/steps/__init__.py.
pytest always imports a directory's conftest.py before collecting any test
module in that directory, regardless of file name or collection order, so
mirroring the same preload list here guarantees every one of these
packages is already cached in sys.modules -- and therefore never "the
first import after the JVM starts" -- no matter which test file happens to
be the one that triggers _init_jvm() first.
"""
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
