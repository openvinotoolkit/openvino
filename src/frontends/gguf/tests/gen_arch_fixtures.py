# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Produce the per-architecture GGUF conversion fixtures consumed by test_arch_conversion.cpp.

A fixture is a GGUF *header*: everything before min(tensor.data_offset), i.e. the magic, the KV
metadata and the tensor table.  That is the entire input to architecture detection and graph
construction -- the weight bytes that follow contribute nothing to the shape of the converted model,
so the C++ test rebuilds a loadable .gguf by appending the recorded number of zero bytes.  Headers
are ~25 KB each; the full models they come from are 559 MB.

The upstream source of the models is llama.cpp's `test-llama-archs`, which walks every architecture
it knows and saves a tiny random model for each.  This script wraps it end to end:

    # from a checkout you already have (nothing is cloned):
    python3 gen_arch_fixtures.py --llama-src <llama.cpp>

    # or let it fetch the pinned commit itself (this is what CI does):
    python3 gen_arch_fixtures.py --fetch --out-dir <dir>

llama.cpp is pinned to LLAMA_CPP_COMMIT below and stays there until a fixture refresh is
deliberately made.  The pin is load-bearing, not hygiene: `test-llama-archs` writes whatever KVs
llama.cpp currently defines, so upstream adding one shifts the bytes of *every* fixture at once
(measured: between 86a9c79f8 and a head 13 days later, all 101 headers changed because three
`attention.indexer.*` KVs were added, and two new architectures appeared).  Tracking a moving
upstream would therefore turn an unrelated llama.cpp commit into a red OpenVINO precommit.  Bump the
pin as its own reviewed change, and say so in the commit message so the provenance stays traceable.

The headers are seed-independent -- verified byte-identical for `-s 1` and `-s 999` -- because the
seed only feeds weight values.  Pinning one anyway (below) costs nothing and keeps the run
deterministic.

Requires the `gguf` Python package (llama.cpp's gguf-py) to read back the generated files.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

# Pinned llama.cpp revision.  MUST be a commit on ggml-org/llama.cpp master: --fetch clones from
# there, so a SHA that only exists in a fork makes CI fail with "couldn't find remote ref".
LLAMA_CPP_COMMIT = "86a9c79f866799eb0e7e89c03578ccfbcc5d808e"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
# The generator's own default seed is std::random_device, i.e. non-reproducible.  Pin it: it does not
# affect the header bytes, but it keeps the whole run deterministic.
SEED = 1

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(HERE, "test_data", "arch_fixtures")

MANIFEST_HEADER = """\
# GGUF per-architecture conversion fixtures -- see gen_arch_fixtures.py (generator) and
# test_arch_conversion.cpp (consumer).  Generated from llama.cpp {commit} at seed {seed}.
#
# One line per fixture:
#   <file> <data_bytes> <expectation>
#
# <file>         header-only fixture in this directory; the loadable .gguf is <file> followed by
#                <data_bytes> zero bytes (all fixture tensors are F32, so no dequant is involved).
# <expectation>  what conversion must do, and the test asserts exactly this:
#   convert   converts cleanly.  The graph fingerprint is pinned in test_arch_conversion.cpp.
#   reject    the architecture is not on the builder's accept list, so conversion must fail with
#             that specific diagnostic -- not a crash, and not a silent wrong-graph success.
#   broken    a supported architecture that currently fails to convert: a real defect, recorded so
#             it cannot be forgotten.  The test asserts it STILL FAILS; fixing the defect turns this
#             line into a failure telling you to promote it to `convert`.
"""


def run(cmd, **kwargs):
    """Run a command, echoing it first so a CI log shows exactly what happened."""
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def fetch_llama_cpp(dest):
    """Shallow-fetch the pinned commit into dest.  ~35 MB, a few seconds.

    A bare `clone --depth 1` cannot take a SHA, and a full clone of llama.cpp is large and slow, so
    fetch the one commit explicitly: init + fetch --depth 1 <sha>.  This requires the SHA to be
    reachable on the remote, which is why LLAMA_CPP_COMMIT must be an upstream commit.
    """
    os.makedirs(dest, exist_ok=True)
    run(["git", "init", "-q", dest])
    run(["git", "-C", dest, "remote", "add", "origin", LLAMA_CPP_REPO])
    run(["git", "-C", dest, "fetch", "-q", "--depth", "1", "origin", LLAMA_CPP_COMMIT])
    run(["git", "-C", dest, "checkout", "-q", "FETCH_HEAD"])
    return dest


def build_generator(src, build_dir, jobs):
    """Configure and build only the test-llama-archs target.

    LLAMA_BUILD_TESTS=ON is required -- the target is a test, and upstream's own release builds turn
    tests off, which is why no prebuilt llama.cpp package ships it.  Everything else is off: the
    examples, tools and server are not needed to save a model.
    """
    run([
        "cmake", "-S", src, "-B", build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=ON",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
    ])
    cmd = ["cmake", "--build", build_dir, "--target", "test-llama-archs"]
    if jobs:
        cmd += ["-j", str(jobs)]
    run(cmd)
    exe = os.path.join(build_dir, "bin", "test-llama-archs")
    if not os.path.isfile(exe):
        sys.exit(f"error: build succeeded but {exe} is missing")
    return exe


def run_generator(exe, out_dir):
    run([exe, "-s", str(SEED), "-o", out_dir], stdout=subprocess.DEVNULL)
    files = sorted(glob.glob(os.path.join(out_dir, "*.gguf")))
    if not files:
        sys.exit("error: generator produced no .gguf files")
    return files


def strip_to_header(path, out_path):
    """Write everything before the first tensor's data offset; return (header_len, data_len, arch)."""
    import gguf

    reader = gguf.GGUFReader(path)
    if not reader.tensors:
        return None
    header_len = min(t.data_offset for t in reader.tensors)
    total = os.path.getsize(path)
    with open(path, "rb") as src, open(out_path, "wb") as dst:
        dst.write(src.read(header_len))
    arch_field = reader.fields["general.architecture"]
    arch = str(bytes(arch_field.parts[-1]), "utf-8")
    return header_len, total - header_len, arch


def read_expectations(manifest):
    """Existing <file> -> <expectation> mapping, or {} if there is no manifest yet."""
    previous = {}
    if not os.path.exists(manifest):
        return previous
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                previous[parts[0]] = parts[2]
    return previous


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--llama-src", help="existing llama.cpp checkout to build from")
    src.add_argument("--fetch", action="store_true",
                     help=f"shallow-fetch the pinned commit ({LLAMA_CPP_COMMIT[:12]}) into a temp dir")
    ap.add_argument("--out-dir", default=FIXTURE_DIR,
                    help="where to write the .gguf.hdr fixtures and manifest.txt (default: the "
                         "in-tree test_data/arch_fixtures)")
    ap.add_argument("--llama-build", help="reuse this build directory instead of a temp one")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="parallel build jobs")
    ap.add_argument("--keep", action="store_true", help="keep the full generated .gguf files")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="gguf_arch_fixtures_")
    try:
        src_dir = args.llama_src
        if args.fetch:
            src_dir = fetch_llama_cpp(os.path.join(work, "llama.cpp"))
        else:
            # An arbitrary checkout may sit at any revision, and the fixture bytes depend on it.
            # Report what we actually built so a mismatch with the pin is visible in the log.
            rev = subprocess.run(["git", "-C", src_dir, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
            if rev and rev != LLAMA_CPP_COMMIT:
                print(f"warning: {src_dir} is at {rev[:12]}, not the pinned {LLAMA_CPP_COMMIT[:12]}; "
                      f"the fixtures will differ from the committed ones", flush=True)

        exe = build_generator(src_dir, args.llama_build or os.path.join(work, "build"), args.jobs)

        models = os.path.join(work, "models")
        os.makedirs(models, exist_ok=True)
        files = run_generator(exe, models)
        print(f"generated {len(files)} models in {models}")

        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        manifest = os.path.join(out_dir, "manifest.txt")
        # Preserve reviewed expectations before the stale fixtures are removed.
        previous = read_expectations(manifest)
        for stale in glob.glob(os.path.join(out_dir, "*.gguf.hdr")):
            os.remove(stale)

        entries = []
        for path in files:
            name = os.path.basename(path) + ".hdr"
            result = strip_to_header(path, os.path.join(out_dir, name))
            if result is None:
                print(f"  skip {os.path.basename(path)}: no tensors")
                continue
            header_len, data_len, arch = result
            entries.append((name, data_len, arch))
            print(f"  {name}: header {header_len} B, data {data_len} B, arch {arch}")

        # Expectations are NOT inferred from the frontend here -- that would make the fixtures assert
        # whatever the frontend currently does, which is not a test.  Preserve the reviewed
        # expectation from the existing manifest and default new fixtures to `reject`, which is
        # correct for any architecture the builder's accept list does not name.  A new architecture
        # that should convert is then a deliberate, reviewable one-word manifest edit.
        with open(manifest, "w") as f:
            f.write(MANIFEST_HEADER.format(commit=LLAMA_CPP_COMMIT, seed=SEED))
            for name, data_len, _arch in sorted(entries):
                expectation = previous.get(name, "reject")
                if name not in previous:
                    print(f"  NEW fixture {name}: defaulted to `reject`; review it")
                f.write(f"{name} {data_len} {expectation}\n")

        total = sum(os.path.getsize(os.path.join(out_dir, n)) for n, _, _ in entries)
        print(f"\nwrote {len(entries)} headers ({total / 1024:.1f} KB raw) + manifest to {out_dir}")

        dropped = set(previous) - {n for n, _, _ in entries}
        if dropped:
            print(f"note: {len(dropped)} fixture(s) disappeared upstream and were dropped: {sorted(dropped)}")
    finally:
        if args.keep:
            print(f"work directory kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
