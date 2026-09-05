from __future__ import annotations

"""
Runtime compatibility patch for a DeepOnto limitation.

DeepOnto's OntologySyntaxParser.parse() (used internally by
OntologyVerbaliser.verbalise_class_expression, which extract.py calls to
turn complex class expressions into natural language) does not support the
OWL `DataHasValue` data property restriction. On any ontology whose complex
class expressions use it -- SNOMED-CT being a real-world example -- it
raises:

    RuntimeError: Input class expression `...` is not in one of the
    supported types.

Confirmed still present on DeepOnto's `main` branch (github.com/KRR-Oxford/DeepOnto)
as of writing -- `OntologySyntaxParser.parse` does no DataHasValue handling
at all, it just abbreviates the expression and parses by parentheses.

Until DeepOnto supports DataHasValue natively, this module monkey-patches
OntologySyntaxParser.parse to drop any `DataHasValue(...)` sub-expression
before parsing, rather than crashing. This is a LOSSY workaround: the
omitted restriction's literal value is dropped from the verbalised text
entirely, not translated into natural language -- a UserWarning is raised
the first time it happens on a given expression so it isn't silently
invisible. If your use case needs that information preserved, this is not
a substitute for a real fix upstream.

extract.py calls apply() once, at import time, before any verbalisation
happens.

This module also starts DeepOnto's JVM as a side effect of being imported,
before it does anything else -- see _init_jvm() below for why that has to
happen here specifically. It also works around two independent, real
segfaults seen on CSF3 when running this project's JVM (via JPype) in the
same process as CUDA (via torch):

  1. _harden_jpype_import_hook() -- JPype's global import hook can
     segfault when asked whether an ordinary (non-Java) Python package is
     secretly a Java package. See that function's docstring.
  2. _init_jvm() starts the JVM with the extra "-Xrs" flag -- a segfault
     was also observed inside JPype's handling of a genuinely-needed Java
     import (`from java.io import File`) itself, which pointed at a
     second, independent cause: HotSpot JVMs install their own SIGSEGV/
     SIGBUS/etc. handlers by default, which is a known source of crashes
     when another native library in the same process (here, CUDA, via
     torch) also relies on signal handling. See _JVM_EXTRA_ARGS below.

Neither of these has been confirmed to fix the crash in the environment
that actually exhibits it -- this codebase's own sandbox has never
reproduced it -- but both target real, documented mechanisms, and are
kept together because they are independent (removing one doesn't affect
the other).
"""

import os
import warnings

import jpype
import jpype.imports  # registers jpype's global import hook as a side effect

import deeponto

_DEFAULT_JVM_MEMORY = "8g"

# Extra flags passed to jpype.startJVM() beyond what deeponto.init_jvm()
# itself uses (just "-ea" and "-Xmx<memory>"). "-Xrs" ("reduce use of OS
# signals") tells the JVM not to install its own handlers for SIGSEGV,
# SIGBUS, SIGQUIT, SIGHUP and SIGTERM. By default, HotSpot JVMs install a
# SIGSEGV/SIGBUS handler as a JIT optimisation (implicit null-pointer
# checks: dereference-and-recover-in-the-handler instead of an explicit
# branch) and handlers for the other three for diagnostics (thread dumps,
# graceful shutdown). Those coexist fine with pure-Python processes, but a
# segfault seen on CSF3 -- in JPype's own, definitely-legitimate handling
# of `from java.io import File` (not some unrelated/misidentified
# package -- see _harden_jpype_import_hook() below for that separate,
# already-fixed issue) -- points at a second, independent problem: the JVM
# sharing a process with CUDA (loaded via torch, confirmed present in the
# crash dump's loaded-extension-modules list) is a known source of signal
# handler conflicts, because CUDA's driver also install its own signal
# handling for its own error reporting. "-Xrs" is the standard mitigation
# for "embedding a JVM in a process that also does its own native signal
# handling" -- it's used for exactly this kind of scenario. Unverified in
# the environment that actually crashes (this sandbox has never
# reproduced the segfault), but it targets a real, documented mechanism
# distinct from (and in addition to) the import-hook hardening below.
_JVM_EXTRA_ARGS = ("-Xrs",)

# The complete set of Java top-level domains anything in genom_pipeline's
# dependency chain actually imports via the `from <java_pkg> import ...`
# syntax -- verified by grepping every .py file in the installed deeponto
# package (not just the ones genom_pipeline currently reaches) for
# `from {java,javax,org,com,de,net,edu,gov,mil,uk}.* import`: only
# `org.*` (OWL API, ELK, HermiT, org.mowl), `java.*` (java.io/java.lang/
# java.util) and `de.tudresden.*` (the jcel reasoner OntologyNormaliser
# uses) are ever used. `javax` is included defensively -- it's one of
# JPype's own standard recognised domains and costs nothing to allow.
_JAVA_TLDS = frozenset({"java", "javax", "org", "de"})


def _harden_jpype_import_hook() -> None:
    """
    Work around a real segfault seen on CSF3 in JPype's global import hook.

    Once the JVM is started, JPype's _JImportLoader.find_spec (registered
    on sys.meta_path by `import jpype.imports` above) is consulted for
    *every* import anywhere in the process, not just Java ones -- and for
    any single-segment module name, its real implementation (see JPype's
    own imports.py) calls into the JVM via `_jpype.isPackage(name)` to ask
    "is this a Java package?", *regardless of whether the name looks like
    one*. On CSF3, asking that question about an ordinary Python package
    name can segfault outright instead of returning False -- three
    different, unrelated packages (torch, transformers's httpx dependency
    pulling in the stdlib mimetypes module, and transformers.trainer_utils
    checking for the optional 'peft' package) were each observed crashing
    this way, at various different points during collection.

    Pre-importing specific heavy packages before the JVM starts (an
    earlier, incomplete version of this fix) only protects against
    whichever package happens to be *first* -- and DeepOnto's own
    deeponto.onto subpackage imports a long, version-dependent tail of
    third-party libraries (click, yacs, rdflib, networkx, nltk, spacy,
    anytree, IPython, dill, PyYAML, plus transformers/torch again) spread
    across its six onto submodules, interleaved with the real Java imports
    those same files need. There's no import order that avoids all of it
    short of enumerating and pre-importing DeepOnto's entire, and possibly
    changing, dependency closure.

    The actual fix is narrower and more robust: wrap find_spec so it only
    ever forwards to JPype's real (crash-prone-for-non-Java-names)
    implementation for names under _JAVA_TLDS above, and returns None
    (meaning "not a Java import, let Python's normal import machinery
    handle it") for everything else -- without ever calling
    `_jpype.isPackage()` on them. This is safe regardless of what
    DeepOnto (or anything else) imports, in what order, or whether the JVM
    has started yet, because nothing in this codebase's dependency chain
    does Java-style imports (`from <pkg> import ...` resolved via this
    hook, as opposed to a normal pip-installed package) outside of
    _JAVA_TLDS -- verified by grepping every .py file in the installed
    deeponto package for such imports.
    """
    loader_cls = jpype.imports._JImportLoader
    if getattr(loader_cls.find_spec, "_genom_hardened", False):
        return

    _original_find_spec = loader_cls.find_spec

    def _hardened_find_spec(self, name, path, target=None):
        if name.partition(".")[0] not in _JAVA_TLDS:
            return None
        return _original_find_spec(self, name, path, target)

    _hardened_find_spec._genom_hardened = True
    loader_cls.find_spec = _hardened_find_spec


_harden_jpype_import_hook()


def _init_jvm() -> None:
    """
    Start the JVM DeepOnto needs (via jpype) before anything imports
    deeponto.onto.

    deeponto/onto/ontology.py runs this at *module import time*:

        if not jpype.isJVMStarted():
            memory = click.prompt("Please enter the maximum memory located "
                                   "to JVM", type=str, default="8g")
            init_jvm(memory)

    i.e. the first time anything anywhere imports deeponto.onto (directly,
    or transitively -- e.g. via deeponto.align.bertmap), DeepOnto blocks on
    an interactive prompt asking how much memory to give the JVM, unless
    the JVM is already started. In any non-interactive context (CI, a
    plain script, pytest with output captured) there's no terminal to
    answer that prompt -- under pytest specifically this fails hard with
    `OSError: reading from stdin while output is captured!` and aborts
    test collection entirely (this is exactly what broke the first real
    CI run: every test module that imports deeponto, directly or
    transitively, errored out with this).

    Calling deeponto.init_jvm() ourselves here -- before any deeponto.onto
    import anywhere in genom_pipeline -- makes `jpype.isJVMStarted()` true
    by the time deeponto/onto/ontology.py's module-level check runs, so it
    skips the prompt entirely. Idempotent: jpype only allows starting the
    JVM once per process, so this is a no-op on any call after the first
    (including a call made from a different module that also imports
    _deeponto_compat).

    Memory limit is controlled by the GENOM_JVM_MEMORY environment
    variable; defaults to "8g" to match DeepOnto's own default.

    Starts the JVM with the extra "-Xrs" flag instead of calling
    deeponto.init_jvm(memory) directly (which starts it without that
    flag) -- see _JVM_EXTRA_ARGS below for why.
    """
    if jpype.isJVMStarted():
        return
    memory = os.environ.get("GENOM_JVM_MEMORY", _DEFAULT_JVM_MEMORY)

    jars_dir = os.path.join(os.path.dirname(os.path.realpath(deeponto.__file__)), "lib")
    if not os.path.isdir(jars_dir):
        # DeepOnto's packaged jars moved or aren't where expected -- fall
        # back to letting DeepOnto start its own JVM rather than fail
        # outright. This loses the -Xrs mitigation below, but only in a
        # situation this module can't handle anyway.
        deeponto.init_jvm(memory)
        return

    jars = os.pathsep.join(os.path.join(jars_dir, name) for name in os.listdir(jars_dir))
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        "-ea",
        *_JVM_EXTRA_ARGS,
        f"-Xmx{memory}",
        "-Djava.class.path=" + jars,
        convertStrings=False,
    )


_init_jvm()

from deeponto.onto.verbalisation import OntologySyntaxParser  # noqa: E402  (must come after _init_jvm())

_TARGET = "DataHasValue"


class DataHasValueOmittedWarning(UserWarning):
    """
    Raised when a DataHasValue(...) sub-expression is dropped while
    verbalising an OWL class expression, because DeepOnto's verbaliser does
    not support it. See genom_pipeline/_deeponto_compat.py.
    """


def _omit_part(expression: str, target: str, pos: int) -> str:
    """
    Remove one `target(...)`-shaped sub-expression starting at `pos`, up to
    its first closing parenthesis. Recurses to remove further occurrences.

    Does not handle nested parentheses *inside* the omitted span -- fine
    for DataHasValue, whose argument is a single literal, not a nested
    expression, but would need generalising for a target that can itself
    contain "(".

    Unlike a naive "chop exactly one character before pos" approach, this
    only trims a preceding space (never a structural '(' or other
    character), so parentheses stay balanced even when the omitted term is
    the first child of its parent expression -- e.g.
    "ObjectIntersectionOf(DataHasValue(p "1") B)" correctly becomes
    "ObjectIntersectionOf(B)", not "ObjectIntersectionOf B)".
    """
    before = expression[:pos].rstrip(" ")
    after = expression[pos:]
    close = after.find(")")
    remainder = after[close + 1 :].lstrip(" ")

    if before and not before.endswith("(") and remainder:
        expression = before + " " + remainder
    else:
        expression = before + remainder

    next_pos = expression.find(target)
    if next_pos != -1:
        return _omit_part(expression, target, next_pos)
    return expression


def _patched_parse(self, owl_expression):
    if not isinstance(owl_expression, str):
        owl_expression = str(owl_expression)

    pos = owl_expression.find(_TARGET)
    if pos != -1:
        warnings.warn(
            f"Dropping an unsupported '{_TARGET}' sub-expression while verbalising "
            "an OWL class expression (DeepOnto's verbaliser does not support it). "
            "The affected concept's logical description / definition-generation "
            "input will be missing this restriction. "
            "See genom_pipeline/_deeponto_compat.py.",
            DataHasValueOmittedWarning,
            stacklevel=3,
        )
        owl_expression = _omit_part(owl_expression, _TARGET, pos)

    return _original_parse(self, owl_expression)


_original_parse = OntologySyntaxParser.parse


def apply() -> None:
    """Patch OntologySyntaxParser.parse in place. Idempotent."""
    if OntologySyntaxParser.parse is _patched_parse:
        return
    OntologySyntaxParser.parse = _patched_parse


def unapply() -> None:
    """Restore DeepOnto's original parse method. Mainly useful for tests."""
    OntologySyntaxParser.parse = _original_parse
