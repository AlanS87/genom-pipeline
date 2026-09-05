"""
pytest configuration for the genom-pipeline test suite.

Import-order fix for a real segfault seen on CSF3: once
genom_pipeline._deeponto_compat starts the JVM (jpype.isJVMStarted() ->
deeponto.init_jvm()), JPype installs a global sys.meta_path import hook
that gets consulted for *every* import anywhere in the process from then
on, not just Java ones. JPype's own docs warn that if a heavy native
extension is imported for the *first* time after that point, resolving one
of its submodules through that hook can segfault outright rather than
raise a catchable exception -- which is exactly what happened:

    File ".../jpype/imports.py", line 174, in find_spec
    File ".../torch/cuda/__init__.py", line 68, in <module>
    File ".../torch/__init__.py", line 2585, in <module>
    File "tests/test_define_batching.py", line 25, in <module>
    Fatal Python error: Segmentation fault

tests/test_deeponto_compat.py imports genom_pipeline._deeponto_compat
directly (to test it), starting the JVM without ever importing torch
first -- _deeponto_compat.py itself never touches torch. Whichever test
file pytest happens to collect next that does `import torch` for the
first time then hits the crash. This bit tests/test_define_batching.py
specifically because it sorts alphabetically right after
test_deeponto_compat.py, so the JVM was already up by the time it was
collected; a different test file ordering (or a different OS/filesystem's
directory listing order) could just as easily have hit it somewhere else.

conftest.py is imported by pytest before it collects any test module in
this directory, so importing torch here first guarantees torch (including
its torch.cuda submodule) is fully loaded and cached in sys.modules
*before* any test file gets a chance to start the JVM. Every later
`import torch` anywhere in the test run then just hits the sys.modules
cache and never reaches JPype's import hook again, regardless of which
test file runs in what order.

This is not a bug in the actual pipeline: genom_pipeline/steps/extract.py
already does `import torch` before `from genom_pipeline import
_deeponto_compat`, so a real run_pipeline() call always imports torch
first. Only the test suite could hit this, because
test_deeponto_compat.py starts the JVM directly without going through
extract.py (or anything else that imports torch) first.
"""
import torch  # noqa: F401  (import order is the whole point here, not usage)
