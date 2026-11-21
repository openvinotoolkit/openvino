# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Pre-codegen include verifier for experimental CM kernels.
#
# Before kernels_db_gen.py inlines every local `#include "foo.hpp"` into the generated
# experimental_cm_kernel_sources.inc, this script statically proves that every referenced
# local header can actually be resolved using the exact same search order the generator uses
# (see Code2CHeaders in graph/common_utils/kernels_db_gen.py):
#
#   kernels_dir
#   headers_dir
#   os.path.join(headers_dir, "../")
#   os.path.join(headers_dir, "batch_headers")
#
# If a `#include "..."` cannot be found in any of those search paths, the build must fail
# loudly (with the include chain that led to the missing header) rather than produce a .inc
# that silently omits the inlined header. This is what B2 in the Fix Plan addresses: a codegen
# produce that "succeeds" but drops a header because its path was never staged.
#
# --self-test constructs throwaway directory fixtures and asserts all three behaviors without
# touching any real kernel/header tree:
#   a) a missing local include fails the verifier (non-zero exit)
#   b) a clean valid include tree passes (exit 0)
#   c) recursive include resolution is followed and a missing transitive header is reported
#      with the full dependency chain

import argparse
import os
import re
import subprocess
import sys
import tempfile

# Local include directives (exactly the shape kernels_db_gen.py's replace_include handles):
#   #include "foo.hpp"      (optionally with a trailing [[no_opt]] attribute).
INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"(\s+\[\[no_opt\]\])?')


def build_search_paths(kernels_dir, headers_dir):
    """Return the search paths in the same order kernels_db_gen.py resolves them."""
    return [
        os.path.abspath(kernels_dir),
        os.path.abspath(headers_dir),
        os.path.abspath(os.path.join(headers_dir, "..")),
        os.path.abspath(os.path.join(headers_dir, "batch_headers")),
    ]


def resolve_include(include_file, search_paths):
    """Return the absolute path of a resolved local header, or None if not found."""
    for path in search_paths:
        candidate = os.path.join(path, include_file)
        if os.path.isfile(candidate):
            return candidate
    return None


def verify_file(filepath, search_paths, chain, errors):
    """Recursively verify all local includes reachable from `filepath`."""
    # Extract all local (double-quoted) includes in this file; angle-bracket includes
    # (`#include <...>`) are system/toolchain headers and are intentionally ignored.
    includes = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except OSError as exc:
        errors.append("  unable to read {}: {}".format(filepath, exc))
        return
    for match in INCLUDE_RE.finditer(content):
        includes.append(match.group(1))

    for include_file in includes:
        resolved = resolve_include(include_file, search_paths)
        resolved_abs = os.path.abspath(resolved) if resolved else None
        # Avoid infinite recursion on circular includes.
        if resolved_abs in chain:
            continue
        if resolved is None:
            errors.append("  {} -> missing local include: {}".format(
                os.path.relpath(filepath), include_file))
            print_chain(chain, filepath, include_file)
            continue
        next_chain = chain + [resolved_abs]
        verify_file(resolved_abs, search_paths, next_chain, errors)


def print_chain(chain, culprit, missing):
    """Print the include chain from the entry point down to the missing header."""
    rel_chain = [os.path.relpath(p) for p in chain]
    print("    include chain: {} -> {} (MISSING)".format(" -> ".join(rel_chain), missing))


def collect_candidates(kernels_dir, headers_dir):
    """Return all kernel (*.cm) and header (*.hpp, *.h) files to verify."""
    files = []
    for root, _, names in os.walk(kernels_dir):
        for name in names:
            if name.endswith('.cm'):
                files.append(os.path.join(root, name))
    for root, _, names in os.walk(headers_dir):
        for name in names:
            if name.endswith('.hpp') or name.endswith('.h'):
                files.append(os.path.join(root, name))
    return sorted(files)


def verify(kernels_dir, headers_dir):
    """Verify all .cm kernels and .hpp/.h headers resolve their local includes."""
    search_paths = build_search_paths(kernels_dir, headers_dir)
    errors = []
    candidates = collect_candidates(kernels_dir, headers_dir)
    for filepath in candidates:
        verify_file(filepath, search_paths, [os.path.abspath(filepath)], errors)
    return errors


def run_self_test():
    """Exercise the verifier's real code path against throwaway fixtures and return the exit code."""
    selftest_failures = []
    with tempfile.TemporaryDirectory() as tmp:
        kernels_dir = os.path.join(tmp, "kernels")
        headers_dir = os.path.join(tmp, "headers")
        kernel_dir = os.path.join(kernels_dir, "kernels")
        os.makedirs(kernel_dir)
        os.makedirs(headers_dir)

        # Valid fixture layout, exercising a transitive include that resolves via
        # headers_dir/.. (the `common/` sibling):
        #   kernels/kernels/main.cm            -> #include "cm_sdpa_common.hpp"
        #   headers/cm_sdpa_common.hpp         -> #include "common/cm_qq_bias_mask.hpp"
        #   headers/common/cm_qq_bias_mask.hpp -> #include <cm/cm.h>
        os.makedirs(os.path.join(headers_dir, "common"))
        with open(os.path.join(kernel_dir, "main.cm"), 'w') as f:
            f.write('#include "cm_sdpa_common.hpp"\n')
        with open(os.path.join(headers_dir, "cm_sdpa_common.hpp"), 'w') as f:
            f.write('#include "common/cm_qq_bias_mask.hpp"\n')
        with open(os.path.join(headers_dir, "common", "cm_qq_bias_mask.hpp"), 'w') as f:
            f.write('#include <cm/cm.h>\n')

        script = os.path.abspath(__file__)

        # (a) Missing local include MUST fail with a non-zero exit code.
        with open(os.path.join(headers_dir, "cm_sdpa_common.hpp"), 'w') as f:
            f.write('#include "does_not_exist.hpp"\n')
        proc_a = subprocess.run([sys.executable, script,
                                 '-in_kernels_dir', kernels_dir,
                                 '-in_headers_dir', headers_dir],
                                capture_output=True, text=True)
        if proc_a.returncode == 0:
            selftest_failures.append("missing-include: expected non-zero exit, got 0")
        # Restore the valid fixture.
        with open(os.path.join(headers_dir, "cm_sdpa_common.hpp"), 'w') as f:
            f.write('#include "common/cm_qq_bias_mask.hpp"\n')

        # (b) Clean valid include resolution MUST exit 0.
        proc_b = subprocess.run([sys.executable, script,
                                 '-in_kernels_dir', kernels_dir,
                                 '-in_headers_dir', headers_dir],
                                capture_output=True, text=True)
        if proc_b.returncode != 0:
            selftest_failures.append("clean-resolve: expected exit 0, got {}".format(proc_b.returncode))

        # (c) Recursive include resolution: a missing TRANSITIVE header must be reported
        #     with the full chain and fail the verification.
        with open(os.path.join(headers_dir, "common", "cm_qq_bias_mask.hpp"), 'w') as f:
            f.write('#include "still_missing.hpp"\n')
        proc_c = subprocess.run([sys.executable, script,
                                 '-in_kernels_dir', kernels_dir,
                                 '-in_headers_dir', headers_dir],
                                capture_output=True, text=True)
        if proc_c.returncode == 0:
            selftest_failures.append("recursive-include: expected non-zero exit, got 0")
        elif "main.cm" not in proc_c.stdout and "cm_sdpa_common.hpp" not in proc_c.stdout:
            selftest_failures.append("recursive-include: missing include chain was not reported")
        # restore
        with open(os.path.join(headers_dir, "common", "cm_qq_bias_mask.hpp"), 'w') as f:
            f.write('#include <cm/cm.h>\n')

    if selftest_failures:
        for failure in selftest_failures:
            print("SELF-TEST-FAIL: {}".format(failure))
        return 1
    print("verify_cm_includes self-test passed: missing-local-include, clean-resolve, and recursive-chain detection all healthy")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Statically verify local include resolution for experimental CM kernels")
    ap.add_argument('-in_kernels_dir', metavar='PATH', help='Absolute path to the staged kernels directory')
    ap.add_argument('-in_headers_dir', metavar='PATH', help='Absolute path to the staged headers root directory')
    ap.add_argument('--self-test', action='store_true', help='Run self-test fixtures instead of a real verification')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(run_self_test())

    if not args.in_kernels_dir or not args.in_headers_dir:
        ap.error('both -in_kernels_dir and -in_headers_dir are required outside --self-test')

    errors = verify(args.in_kernels_dir, args.in_headers_dir)
    if errors:
        print("verify_cm_includes: FAILED to resolve {} local include(s):".format(len(errors)))
        for error in errors:
            print(error)
        sys.exit(1)
    print("verify_cm_includes: all local includes resolved cleanly")


if __name__ == "__main__":
    main()