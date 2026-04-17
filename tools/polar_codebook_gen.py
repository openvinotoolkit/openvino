# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
PolarQuant codebook generator.

Generates Lloyd-Max optimal codebooks for polar angle distributions at each
tree level, along with precomputed cos/sin lookup tables. Outputs a C header
file (polarq_tables.h) suitable for inclusion in the OpenVINO CPU plugin.

Usage:
    python tools/polar_codebook_gen.py [--samples N] [--output PATH]

Algorithm:
    1. Generate N random dim-dimensional Gaussian vectors
    2. Apply Haar random orthogonal rotation (same seed as C++ code)
    3. Recursive polar decomposition: pairs → (angle, radius) at each level
    4. Fit Lloyd-Max quantizer per (level, bit-width) pair
    5. Compute cos/sin LUTs from codebook centroids
    6. Export as C constexpr arrays
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# Must match TURBOQ_SEED in turboq_rotation.cpp
TURBOQ_SEED = 0x517CC1B727220A95


def haar_rotation_matrix(dim: int, seed: int = TURBOQ_SEED) -> np.ndarray:
    """Generate a Haar-distributed random orthogonal matrix via QR of Gaussian."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((dim, dim)).astype(np.float64)
    Q, R = np.linalg.qr(Z)
    # Ensure uniqueness: multiply columns by sign of diagonal of R
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    Q = Q * d[np.newaxis, :]
    return Q


def polar_decompose_angles(rotated: np.ndarray) -> list[np.ndarray]:
    """
    Recursive polar decomposition of rotated vectors.

    Args:
        rotated: (N, dim) array of rotated vectors

    Returns:
        List of L arrays, where angles[k] has shape (N, dim // 2^(k+1))
        Level 0 (k=0): angles in [0, 2*pi)  (level 1 in the spec)
        Level k>0: angles in [0, pi/2]
    """
    N, dim = rotated.shape
    L = int(np.log2(dim))
    assert 2**L == dim, f"dim={dim} must be power of 2"

    angles = []
    radii = rotated.copy()

    for level in range(L):
        n_pairs = radii.shape[1] // 2
        even = radii[:, 0::2]  # (N, n_pairs)
        odd = radii[:, 1::2]   # (N, n_pairs)

        if level == 0:
            # Level 1: atan2 in [0, 2*pi)
            ang = np.arctan2(odd, even)
            ang = np.mod(ang, 2 * np.pi)
        else:
            # Level k>1: atan2 in [0, pi/2] (both radii are non-negative)
            ang = np.arctan2(odd, even)

        new_radii = np.sqrt(even**2 + odd**2)
        angles.append(ang)
        radii = new_radii

    return angles


def lloyd_max(samples: np.ndarray, n_levels: int, max_iter: int = 200,
              tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Lloyd-Max scalar quantizer.

    Args:
        samples: 1-D array of samples
        n_levels: number of quantization levels (2^bits)
        max_iter: maximum iterations
        tol: convergence tolerance (relative MSE change)

    Returns:
        (centroids, boundaries, mse)
        centroids: (n_levels,) sorted array
        boundaries: (n_levels-1,) decision boundaries
        mse: mean squared error
    """
    # Initialize with uniform quantiles
    percentiles = np.linspace(0, 100, n_levels + 1)
    boundaries = np.percentile(samples, percentiles[1:-1])
    centroids = np.zeros(n_levels)

    prev_mse = np.inf
    for _ in range(max_iter):
        # Assign samples to nearest centroid region
        assignments = np.digitize(samples, boundaries)

        # Update centroids
        for i in range(n_levels):
            mask = assignments == i
            if np.any(mask):
                centroids[i] = np.mean(samples[mask])
            else:
                # Empty bin: keep previous or use boundary midpoint
                if i == 0:
                    centroids[i] = boundaries[0] - 0.1
                elif i == n_levels - 1:
                    centroids[i] = boundaries[-1] + 0.1
                else:
                    centroids[i] = (boundaries[i - 1] + boundaries[i]) / 2

        # Update boundaries (midpoints)
        boundaries = (centroids[:-1] + centroids[1:]) / 2

        # Check convergence
        reconstructed = centroids[np.digitize(samples, boundaries)]
        mse = np.mean((samples - reconstructed) ** 2)
        if abs(prev_mse - mse) / max(prev_mse, 1e-30) < tol:
            break
        prev_mse = mse

    return centroids, boundaries, mse


def generate_codebooks(n_samples: int = 10_000_000,
                       dims: list[int] | None = None,
                       bit_widths: list[int] | None = None,
                       ) -> dict:
    """
    Generate polar angle codebooks for all levels and bit widths.

    Returns dict mapping (level_1based, bits) -> {centroids, boundaries, cos_lut, sin_lut, mse}
    Level k's distribution depends on 2^(k-1) effective d.f., not dim.
    We validate this by checking consistency across dims.
    """
    if dims is None:
        dims = [128, 256]
    if bit_widths is None:
        bit_widths = [2, 3, 4, 5]

    max_dim = max(dims)
    max_L = int(np.log2(max_dim))
    print(f"Generating {n_samples} samples with dim={max_dim} (L={max_L})...", file=sys.stderr)

    # Generate random Gaussian vectors
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_samples, max_dim)).astype(np.float64)

    # Normalize to unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-30, 1.0, norms)
    unit = vectors / norms

    # Apply rotation and scale by sqrt(dim)
    Q = haar_rotation_matrix(max_dim)
    sqrt_dim = np.sqrt(float(max_dim))
    rotated = (unit @ Q.T) * sqrt_dim
    print("Rotation applied.", file=sys.stderr)

    # Polar decomposition
    angles = polar_decompose_angles(rotated)
    print(f"Polar decomposition complete: {len(angles)} levels.", file=sys.stderr)

    codebooks = {}
    for level_0 in range(max_L):
        level_1 = level_0 + 1  # 1-based level
        all_angles = angles[level_0].flatten()

        for bits in bit_widths:
            n_levels = 2**bits
            print(f"  Level {level_1}, {bits}-bit ({n_levels} levels): "
                  f"{len(all_angles)} samples...", end="", file=sys.stderr)

            centroids, boundaries, mse = lloyd_max(all_angles, n_levels)

            # Precompute cos/sin LUTs
            cos_lut = np.cos(centroids).astype(np.float32)
            sin_lut = np.sin(centroids).astype(np.float32)

            codebooks[(level_1, bits)] = {
                "centroids": centroids.astype(np.float32),
                "boundaries": boundaries.astype(np.float32),
                "cos_lut": cos_lut,
                "sin_lut": sin_lut,
                "mse": float(mse),
                "n_samples": len(all_angles),
            }
            variance = np.var(all_angles)
            print(f" MSE={mse:.6f}, var={variance:.6f}, SNR={10*np.log10(variance/max(mse,1e-30)):.1f} dB",
                  file=sys.stderr)

    return codebooks


def format_float_array(arr: np.ndarray, name: str, indent: str = "    ") -> str:
    """Format a float array as a C constexpr initializer."""
    n = len(arr)
    lines = [f"static constexpr float {name}[{n}] = {{"]
    for i in range(0, n, 4):
        chunk = arr[i:min(i + 4, n)]
        vals = ", ".join(f"{v:14.8f}f" for v in chunk)
        suffix = "," if i + 4 < n else ""
        lines.append(f"{indent}{vals}{suffix}")
    lines.append("};")
    return "\n".join(lines)


def generate_header(codebooks: dict, output_path: str) -> None:
    """Generate the polarq_tables.h C header file."""

    # Bit allocations from the implementation plan
    # polar4 (dim=128): L1=4, L2=5, L3=4, L4=3, L5=2, L6=0, L7=0
    # polar3 (dim=128): L1=3, L2=3, L3=3, L4=3, L5=2, L6=2, L7=0
    polar4_bits = {1: 4, 2: 5, 3: 4, 4: 3, 5: 2, 6: 0, 7: 0, 8: 0}
    polar3_bits = {1: 3, 2: 3, 3: 3, 4: 3, 5: 2, 6: 2, 7: 0, 8: 0}

    lines = [
        "// Copyright (C) 2018-2026 Intel Corporation",
        "// SPDX-License-Identifier: Apache-2.0",
        "//",
        "",
        "#pragma once",
        "",
        "// Auto-generated by tools/polar_codebook_gen.py — do not edit manually.",
        "// PolarQuant per-level angle codebooks, decision boundaries, and cos/sin LUTs.",
        "",
        "#include <cmath>",
        "#include <cstddef>",
        "",
    ]

    # Emit codebook tables for each (level, bits) pair that's actually used
    used_pairs = set()
    for level in range(1, 9):
        for alloc in [polar4_bits, polar3_bits]:
            b = alloc.get(level, 0)
            if b > 0:
                used_pairs.add((level, b))

    # Sort for deterministic output
    for level, bits in sorted(used_pairs):
        key = (level, bits)
        if key not in codebooks:
            print(f"WARNING: no codebook for level={level}, bits={bits}", file=sys.stderr)
            continue

        cb = codebooks[key]
        tag = f"L{level}_{bits}BIT"

        lines.append(f"// Level {level}, {bits}-bit: {2**bits} centroids")
        lines.append(f"// MSE={cb['mse']:.6f}, samples={cb['n_samples']}")
        lines.append(format_float_array(cb["centroids"], f"POLARQ_CENTROIDS_{tag}"))
        lines.append("")
        lines.append(format_float_array(cb["boundaries"], f"POLARQ_BOUNDARIES_{tag}"))
        lines.append("")
        lines.append(format_float_array(cb["cos_lut"], f"POLARQ_COS_{tag}"))
        lines.append("")
        lines.append(format_float_array(cb["sin_lut"], f"POLARQ_SIN_{tag}"))
        lines.append("")

    # Bit allocation structs
    lines.append("// -----------------------------------------------------------------------")
    lines.append("// Bit allocation tables.")
    lines.append("// -----------------------------------------------------------------------")
    lines.append("")
    lines.append("// Maximum tree depth supported (dim=256 → L=8).")
    lines.append("static constexpr int POLARQ_MAX_LEVELS = 8;")
    lines.append("")

    def emit_alloc(name: str, alloc: dict, max_level: int) -> None:
        vals = ", ".join(str(alloc.get(i, 0)) for i in range(1, max_level + 1))
        lines.append(f"// Bits per angle at each level (1-indexed: [0]=level1, [1]=level2, ...).")
        lines.append(f"static constexpr int {name}[POLARQ_MAX_LEVELS] = {{{vals}}};")
        lines.append("")

    emit_alloc("POLARQ_BITS_POLAR4", polar4_bits, 8)
    emit_alloc("POLARQ_BITS_POLAR3", polar3_bits, 8)

    # Head byte size computation
    lines.append("// Compute packed head record size in bytes.")
    lines.append("// index_bits = sum over levels of (dim/2^level * bits_per_level), + 32 bits for fp32 norm.")
    lines.append("static inline size_t polarq_head_bytes(int head_dim, const int* bits_per_level) {")
    lines.append("    int total_bits = 0;")
    lines.append("    int dim = head_dim;")
    lines.append("    int L = 0;")
    lines.append("    for (int d = dim; d > 1; d >>= 1) L++;")
    lines.append("    for (int k = 0; k < L; k++) {")
    lines.append("        int n_angles = dim >> (k + 1);")
    lines.append("        total_bits += n_angles * bits_per_level[k];")
    lines.append("    }")
    lines.append("    return static_cast<size_t>((total_bits + 7) / 8) + 4;  // + fp32 norm")
    lines.append("}")
    lines.append("")

    # Convenience: fixed angle value for levels with 0 bits
    lines.append("// Fixed angle for levels with 0 allocated bits (angle = pi/4).")
    lines.append("static constexpr float POLARQ_FIXED_ANGLE = 0.78539816339744830962f;  // pi/4")
    lines.append("static constexpr float POLARQ_FIXED_COS = 0.70710678118654752440f;    // cos(pi/4) = 1/sqrt(2)")
    lines.append("static constexpr float POLARQ_FIXED_SIN = 0.70710678118654752440f;    // sin(pi/4) = 1/sqrt(2)")
    lines.append("")

    # LUT accessor helpers
    lines.append("// -----------------------------------------------------------------------")
    lines.append("// LUT accessor: returns pointers to centroids/boundaries/cos/sin for a")
    lines.append("// given (level, bits) pair.  Returns nullptr for 0-bit levels.")
    lines.append("// -----------------------------------------------------------------------")
    lines.append("")

    # Generate a struct to hold pointers
    lines.append("struct PolarqLevelLUT {")
    lines.append("    const float* centroids;")
    lines.append("    const float* boundaries;")
    lines.append("    const float* cos_lut;")
    lines.append("    const float* sin_lut;")
    lines.append("    int n_centroids;  // 2^bits, or 0 if fixed")
    lines.append("};")
    lines.append("")

    # Generate the accessor function
    lines.append("static inline PolarqLevelLUT polarq_get_lut(int level_1based, int bits) {")
    lines.append("    if (bits == 0) return {nullptr, nullptr, nullptr, nullptr, 0};")
    lines.append("    switch (level_1based * 16 + bits) {")
    for level, bits in sorted(used_pairs):
        key = (level, bits)
        if key not in codebooks:
            continue
        tag = f"L{level}_{bits}BIT"
        n = 2**bits
        lines.append(f"    case {level * 16 + bits}: return {{POLARQ_CENTROIDS_{tag}, POLARQ_BOUNDARIES_{tag}, "
                     f"POLARQ_COS_{tag}, POLARQ_SIN_{tag}, {n}}};")
    lines.append("    default: return {nullptr, nullptr, nullptr, nullptr, 0};")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # Write output
    header_text = "\n".join(lines) + "\n"
    Path(output_path).write_text(header_text)
    print(f"Wrote {output_path} ({len(header_text)} bytes)", file=sys.stderr)


def print_mse_report(codebooks: dict) -> None:
    """Print per-level MSE analysis for bit allocation optimization."""
    print("\n=== Per-Level MSE Report ===")
    print(f"{'Level':>6} {'Bits':>5} {'Levels':>7} {'MSE':>12} {'Variance':>12} {'SNR(dB)':>10}")
    print("-" * 60)

    for (level, bits), cb in sorted(codebooks.items()):
        n_levels = 2**bits
        mse = cb["mse"]
        # Compute variance from samples (stored in centroids shape gives us n_levels)
        snr = 10 * np.log10(1.0 / max(mse, 1e-30)) if mse > 0 else float("inf")
        print(f"{level:>6} {bits:>5} {n_levels:>7} {mse:>12.6f} {'—':>12} {snr:>10.1f}")


def main():
    parser = argparse.ArgumentParser(description="PolarQuant codebook generator")
    parser.add_argument("--samples", type=int, default=2_000_000,
                        help="Number of random vectors to generate (default: 2M)")
    parser.add_argument("--output", type=str,
                        default="src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/polarq_tables.h",
                        help="Output header file path")
    parser.add_argument("--report-only", action="store_true",
                        help="Only print MSE report, don't write header")
    args = parser.parse_args()

    codebooks = generate_codebooks(n_samples=args.samples)
    print_mse_report(codebooks)

    if not args.report_only:
        generate_header(codebooks, args.output)


if __name__ == "__main__":
    main()
