// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Generates per-op reference data for the GGUF frontend tests by running the ops
// through *real ggml* (the same code llama.cpp executes), NOT a numpy reimplementation
// of the formula. This is the authoritative oracle: it catches translator/formula
// mismatches (e.g. GELU erf-vs-tanh) that a hand-rolled numpy reference would encode
// and therefore hide.
//
// For each op it writes <name>_input*.bin and <name>_expected.bin as raw little-endian
// f32 (shape recorded in a sidecar <name>.shape text file). A small python step
// (gen_ggml_reference.py) converts these to the .npy files the C++ tests load, so the
// committed test vectors are byte-for-byte ggml outputs.
//
// Build & run (see gen_ggml_reference.py which wraps this):
//   cc gen_ggml_reference.c -I<llama>/ggml/include -L<llama>/build-ref/bin -lggml -lggml-base -lggml-cpu -o gen_ggml_reference
//   LD_LIBRARY_PATH=<llama>/build-ref/bin ./gen_ggml_reference <out_dir>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

static void write_shape(const char* dir, const char* name, const int64_t* dims, int ndim) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.shape", dir, name);
    FILE* f = fopen(path, "w");
    for (int i = 0; i < ndim; ++i) fprintf(f, "%lld%s", (long long)dims[i], i + 1 < ndim ? " " : "");
    fprintf(f, "\n");
    fclose(f);
}

static void write_bin(const char* dir, const char* name, const float* data, int64_t n,
                      const int64_t* dims, int ndim) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fwrite(data, sizeof(float), (size_t)n, f);
    fclose(f);
    write_shape(dir, name, dims, ndim);
}

// Raw byte dump (for quantized blocks). Shape file records the byte count.
static void write_raw(const char* dir, const char* name, const void* data, int64_t nbytes) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fwrite(data, 1, (size_t)nbytes, f);
    fclose(f);
    int64_t d[1] = { nbytes };
    write_shape(dir, name, d, 1);
}

// Build a 1-input unary graph, run it on the CPU backend, dump input + output.
typedef struct ggml_tensor* (*unary_fn)(struct ggml_context*, struct ggml_tensor*);

static void run_unary(const char* dir, const char* name, unary_fn fn,
                      const float* in, const int64_t* dims, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= dims[i];

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, false };
    struct ggml_context* ctx = ggml_init(params);

    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < ndim; ++i) ne[i] = dims[ndim - 1 - i];  // ggml is reversed vs row-major
    struct ggml_tensor* x = ggml_new_tensor(ctx, GGML_TYPE_F32, ndim, ne);
    memcpy(x->data, in, sizeof(float) * (size_t)n);

    struct ggml_tensor* y = fn(ctx, x);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    char inn[256];
    snprintf(inn, sizeof(inn), "%s_input", name);
    char outn[256];
    snprintf(outn, sizeof(outn), "%s_expected", name);
    write_bin(dir, inn, in, n, dims, ndim);
    write_bin(dir, outn, (const float*)y->data, n, dims, ndim);
    printf("  %-16s in/out %lld elems\n", name, (long long)n);

    ggml_free(ctx);
}

// Quantize synthetic float data with ggml, then dump (a) the raw quantized block bytes and
// (b) ggml's own dequantization of those exact bytes. The C++ test feeds the SAME bytes through
// the frontend's dequant subgraph and compares against ggml's to_float -- so ggml is the oracle
// for both the quantization AND the dequantization, exactly like llama.cpp test-quantize-fns.
//
// Emits: <name>_qbytes.bin (u8 raw block) + <name>_qbytes.shape ("nbytes")
//        <name>_deq.bin    (f32 ggml dequant) + <name>_deq.shape ("rows cols")
static void run_dequant(const char* dir, const char* name, enum ggml_type type,
                        int64_t rows, int64_t cols) {
    const int64_t n = rows * cols;
    const struct ggml_type_traits* tt = ggml_get_type_traits(type);
    if (!tt || !tt->to_float) { fprintf(stderr, "no traits for %s\n", name); return; }

    float* in = malloc(sizeof(float) * n);
    for (int64_t i = 0; i < n; ++i) {
        // Smooth, asymmetric data so per-block scale AND min are non-trivial (fractional) --
        // the case that exposes an integer-zero-point dequant bug (Q4_K/Q5_K).
        in[i] = 0.37f + 1.7f * sinf(0.013f * (float)i) + 0.4f * cosf(0.0007f * (float)i * (float)i);
    }

    const size_t row_bytes = ggml_row_size(type, cols);
    uint8_t* q = malloc(row_bytes * rows);
    // quantize row by row (ggml_quantize_chunk works on whole rows)
    ggml_quantize_chunk(type, in, q, 0, rows, cols, NULL);

    float* deq = malloc(sizeof(float) * n);
    for (int64_t r = 0; r < rows; ++r) {
        tt->to_float(q + r * row_bytes, deq + r * cols, cols);
    }

    char nm[256];
    snprintf(nm, sizeof(nm), "%s_qbytes", name);
    write_raw(dir, nm, q, (int64_t)(row_bytes * rows));

    int64_t ddims[2] = { rows, cols };
    snprintf(nm, sizeof(nm), "%s_deq", name);
    write_bin(dir, nm, deq, n, ddims, 2);
    printf("  %-12s rows=%lld cols=%lld qbytes=%lld\n", name, (long long)rows, (long long)cols,
           (long long)(row_bytes * rows));

    free(in); free(q); free(deq);
}

int main(int argc, char** argv) {
    const char* out_dir = argc > 1 ? argv[1] : ".";

    // Deterministic input spanning the interesting activation range.
    const int64_t dims[2] = {4, 32};  // 128 values
    int64_t n = dims[0] * dims[1];
    float* in = malloc(sizeof(float) * n);
    for (int64_t i = 0; i < n; ++i) {
        // range roughly [-6, 6]
        in[i] = -6.0f + 12.0f * (float)i / (float)(n - 1);
    }

    printf("generating ggml reference data into %s\n", out_dir);
    run_unary(out_dir, "gelu_ggml",       ggml_gelu,       in, dims, 2);
    run_unary(out_dir, "gelu_erf_ggml",   ggml_gelu_erf,   in, dims, 2);
    run_unary(out_dir, "gelu_quick_ggml", ggml_gelu_quick, in, dims, 2);
    run_unary(out_dir, "silu_ggml",       ggml_silu,       in, dims, 2);
    free(in);

    // Dequant references for every quant type the GGUF frontend supports. cols=256 satisfies
    // every block/super-block size (32 and 256); 4 rows exercises multiple blocks.
    printf("generating ggml dequant reference data\n");
    const int64_t R = 4, C = 256;
    run_dequant(out_dir, "q4_0", GGML_TYPE_Q4_0, R, C);
    run_dequant(out_dir, "q4_1", GGML_TYPE_Q4_1, R, C);
    run_dequant(out_dir, "q5_0", GGML_TYPE_Q5_0, R, C);
    run_dequant(out_dir, "q5_1", GGML_TYPE_Q5_1, R, C);
    run_dequant(out_dir, "q8_0", GGML_TYPE_Q8_0, R, C);
    run_dequant(out_dir, "q2_k", GGML_TYPE_Q2_K, R, C);
    run_dequant(out_dir, "q3_k", GGML_TYPE_Q3_K, R, C);
    run_dequant(out_dir, "q4_k", GGML_TYPE_Q4_K, R, C);
    run_dequant(out_dir, "q5_k", GGML_TYPE_Q5_K, R, C);
    run_dequant(out_dir, "q6_k", GGML_TYPE_Q6_K, R, C);
    run_dequant(out_dir, "q2_0", GGML_TYPE_Q2_0, R, C);

    return 0;
}
