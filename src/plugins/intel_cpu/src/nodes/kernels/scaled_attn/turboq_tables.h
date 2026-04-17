// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>

// Lloyd-Max optimal centroids for the marginal distribution of a rotated
// unit-norm coordinate (scaled by sqrt(d)). This distribution is Beta-derived
// and converges to N(0,1) as d→∞.
//
// Per-dim tables (D128, D256, D512) give slightly better quantization error
// than the asymptotic N(0,1) values, especially for 4-bit quantization.
// Asymptotic values (no suffix) are used as fallback for other head_dims.
//
// Symmetric: codebook[i] == -codebook[2^b - 1 - i].

// ---------------------------------------------------------------------------
// Asymptotic (N(0,1), d→∞) — fallback for head_dims other than 128/256/512.
// ---------------------------------------------------------------------------
static constexpr float TURBOQ_CODEBOOK_2BIT[4] = {
    -1.51040113f,
    -0.45277205f,
    +0.45277205f,
    +1.51040113f,
};

static constexpr float TURBOQ_CODEBOOK_3BIT[8] = {
    -2.15187458f,
    -1.34383723f,
    -0.75595247f,
    -0.24507484f,
    +0.24507484f,
    +0.75595247f,
    +1.34383723f,
    +2.15187458f,
};

static constexpr float TURBOQ_CODEBOOK_4BIT[16] = {
    -2.73235729f,
    -2.06875481f,
    -1.61777864f,
    -1.25597887f,
    -0.94212216f,
    -0.65659074f,
    -0.38794210f,
    -0.12835875f,
    +0.12835875f,
    +0.38794210f,
    +0.65659074f,
    +0.94212216f,
    +1.25597887f,
    +1.61777864f,
    +2.06875481f,
    +2.73235729f,
};

// ---------------------------------------------------------------------------
// head_dim = 128
// ---------------------------------------------------------------------------
static constexpr float TURBOQ_CODEBOOK_2BIT_D128[4] = {
    -1.50517782f,
    -0.45244580f,
    +0.45244580f,
    +1.50517782f,
};

static constexpr float TURBOQ_CODEBOOK_3BIT_D128[8] = {
    -2.13140531f,
    -1.33653124f,
    -0.75328017f,
    -0.24440650f,
    +0.24440650f,
    +0.75328017f,
    +1.33653124f,
    +2.13140531f,
};

static constexpr float TURBOQ_CODEBOOK_4BIT_D128[16] = {
    -2.68863793f,
    -2.04567026f,
    -1.60407964f,
    -1.24752097f,
    -0.93688023f,
    -0.65345125f,
    -0.38627496f,
    -0.12783692f,
    +0.12783692f,
    +0.38627496f,
    +0.65345125f,
    +0.93688023f,
    +1.24752097f,
    +1.60407964f,
    +2.04567026f,
    +2.68863793f,
};

// ---------------------------------------------------------------------------
// head_dim = 256
// ---------------------------------------------------------------------------
static constexpr float TURBOQ_CODEBOOK_2BIT_D256[4] = {
    -1.50778871f,
    -0.45260987f,
    +0.45260987f,
    +1.50778871f,
};

static constexpr float TURBOQ_CODEBOOK_3BIT_D256[8] = {
    -2.14160037f,
    -1.34017750f,
    -0.75461598f,
    -0.24474095f,
    +0.24474095f,
    +0.75461598f,
    +1.34017750f,
    +2.14160037f,
};

static constexpr float TURBOQ_CODEBOOK_4BIT_D256[16] = {
    -2.71033623f,
    -2.05714788f,
    -1.61089805f,
    -1.25173367f,
    -0.93949232f,
    -0.65501613f,
    -0.38710608f,
    -0.12809709f,
    +0.12809709f,
    +0.38710608f,
    +0.65501613f,
    +0.93949232f,
    +1.25173367f,
    +1.61089805f,
    +2.05714788f,
    +2.71033623f,
};

// ---------------------------------------------------------------------------
// head_dim = 512
// ---------------------------------------------------------------------------
static constexpr float TURBOQ_CODEBOOK_2BIT_D512[4] = {
    -1.50909473f,
    -0.45269120f,
    +0.45269120f,
    +1.50909473f,
};

static constexpr float TURBOQ_CODEBOOK_3BIT_D512[8] = {
    -2.14672756f,
    -1.34200568f,
    -0.75528415f,
    -0.24490797f,
    +0.24490797f,
    +0.75528415f,
    +1.34200568f,
    +2.14672756f,
};

static constexpr float TURBOQ_CODEBOOK_4BIT_D512[16] = {
    -2.72131117f,
    -2.06294084f,
    -1.61433642f,
    -1.25385775f,
    -0.94080983f,
    -0.65580594f,
    -0.38752582f,
    -0.12822853f,
    +0.12822853f,
    +0.38752582f,
    +0.65580594f,
    +0.94080983f,
    +1.25385775f,
    +1.61433642f,
    +2.06294084f,
    +2.72131117f,
};

// ---------------------------------------------------------------------------
// Decision boundaries = midpoints between adjacent centroids (per-dim).
// ---------------------------------------------------------------------------
#define TBQ_MIDS_2BIT(CB) {((CB)[0] + (CB)[1]) / 2.0f, ((CB)[1] + (CB)[2]) / 2.0f, ((CB)[2] + (CB)[3]) / 2.0f}
#define TBQ_MIDS_3BIT(CB)        \
    {((CB)[0] + (CB)[1]) / 2.0f, \
     ((CB)[1] + (CB)[2]) / 2.0f, \
     ((CB)[2] + (CB)[3]) / 2.0f, \
     ((CB)[3] + (CB)[4]) / 2.0f, \
     ((CB)[4] + (CB)[5]) / 2.0f, \
     ((CB)[5] + (CB)[6]) / 2.0f, \
     ((CB)[6] + (CB)[7]) / 2.0f}
#define TBQ_MIDS_4BIT(CB)          \
    {((CB)[0] + (CB)[1]) / 2.0f,   \
     ((CB)[1] + (CB)[2]) / 2.0f,   \
     ((CB)[2] + (CB)[3]) / 2.0f,   \
     ((CB)[3] + (CB)[4]) / 2.0f,   \
     ((CB)[4] + (CB)[5]) / 2.0f,   \
     ((CB)[5] + (CB)[6]) / 2.0f,   \
     ((CB)[6] + (CB)[7]) / 2.0f,   \
     ((CB)[7] + (CB)[8]) / 2.0f,   \
     ((CB)[8] + (CB)[9]) / 2.0f,   \
     ((CB)[9] + (CB)[10]) / 2.0f,  \
     ((CB)[10] + (CB)[11]) / 2.0f, \
     ((CB)[11] + (CB)[12]) / 2.0f, \
     ((CB)[12] + (CB)[13]) / 2.0f, \
     ((CB)[13] + (CB)[14]) / 2.0f, \
     ((CB)[14] + (CB)[15]) / 2.0f}

static constexpr float TURBOQ_BOUNDARIES_2BIT[3] = TBQ_MIDS_2BIT(TURBOQ_CODEBOOK_2BIT);
static constexpr float TURBOQ_BOUNDARIES_3BIT[7] = TBQ_MIDS_3BIT(TURBOQ_CODEBOOK_3BIT);
static constexpr float TURBOQ_BOUNDARIES_4BIT[15] = TBQ_MIDS_4BIT(TURBOQ_CODEBOOK_4BIT);

static constexpr float TURBOQ_BOUNDARIES_2BIT_D128[3] = TBQ_MIDS_2BIT(TURBOQ_CODEBOOK_2BIT_D128);
static constexpr float TURBOQ_BOUNDARIES_3BIT_D128[7] = TBQ_MIDS_3BIT(TURBOQ_CODEBOOK_3BIT_D128);
static constexpr float TURBOQ_BOUNDARIES_4BIT_D128[15] = TBQ_MIDS_4BIT(TURBOQ_CODEBOOK_4BIT_D128);

static constexpr float TURBOQ_BOUNDARIES_2BIT_D256[3] = TBQ_MIDS_2BIT(TURBOQ_CODEBOOK_2BIT_D256);
static constexpr float TURBOQ_BOUNDARIES_3BIT_D256[7] = TBQ_MIDS_3BIT(TURBOQ_CODEBOOK_3BIT_D256);
static constexpr float TURBOQ_BOUNDARIES_4BIT_D256[15] = TBQ_MIDS_4BIT(TURBOQ_CODEBOOK_4BIT_D256);

static constexpr float TURBOQ_BOUNDARIES_2BIT_D512[3] = TBQ_MIDS_2BIT(TURBOQ_CODEBOOK_2BIT_D512);
static constexpr float TURBOQ_BOUNDARIES_3BIT_D512[7] = TBQ_MIDS_3BIT(TURBOQ_CODEBOOK_3BIT_D512);
static constexpr float TURBOQ_BOUNDARIES_4BIT_D512[15] = TBQ_MIDS_4BIT(TURBOQ_CODEBOOK_4BIT_D512);

#undef TBQ_MIDS_2BIT
#undef TBQ_MIDS_3BIT
#undef TBQ_MIDS_4BIT

// ---------------------------------------------------------------------------
// Lookup: returns the right codebook/boundaries pointer for (bits, head_dim).
// Fallback for unknown head_dim: asymptotic (N(0,1)) values.
// ---------------------------------------------------------------------------
inline const float* turboq_codebook(int bits, int head_dim) {
    switch (head_dim) {
    case 128:
        return bits == 4   ? TURBOQ_CODEBOOK_4BIT_D128
               : bits == 3 ? TURBOQ_CODEBOOK_3BIT_D128
                           : TURBOQ_CODEBOOK_2BIT_D128;
    case 256:
        return bits == 4   ? TURBOQ_CODEBOOK_4BIT_D256
               : bits == 3 ? TURBOQ_CODEBOOK_3BIT_D256
                           : TURBOQ_CODEBOOK_2BIT_D256;
    case 512:
        return bits == 4   ? TURBOQ_CODEBOOK_4BIT_D512
               : bits == 3 ? TURBOQ_CODEBOOK_3BIT_D512
                           : TURBOQ_CODEBOOK_2BIT_D512;
    default:
        return bits == 4 ? TURBOQ_CODEBOOK_4BIT : bits == 3 ? TURBOQ_CODEBOOK_3BIT : TURBOQ_CODEBOOK_2BIT;
    }
}

inline const float* turboq_boundaries(int bits, int head_dim) {
    switch (head_dim) {
    case 128:
        return bits == 4   ? TURBOQ_BOUNDARIES_4BIT_D128
               : bits == 3 ? TURBOQ_BOUNDARIES_3BIT_D128
                           : TURBOQ_BOUNDARIES_2BIT_D128;
    case 256:
        return bits == 4   ? TURBOQ_BOUNDARIES_4BIT_D256
               : bits == 3 ? TURBOQ_BOUNDARIES_3BIT_D256
                           : TURBOQ_BOUNDARIES_2BIT_D256;
    case 512:
        return bits == 4   ? TURBOQ_BOUNDARIES_4BIT_D512
               : bits == 3 ? TURBOQ_BOUNDARIES_3BIT_D512
                           : TURBOQ_BOUNDARIES_2BIT_D512;
    default:
        return bits == 4 ? TURBOQ_BOUNDARIES_4BIT : bits == 3 ? TURBOQ_BOUNDARIES_3BIT : TURBOQ_BOUNDARIES_2BIT;
    }
}

// ---------------------------------------------------------------------------
// Other constants.
// ---------------------------------------------------------------------------

// v1 integration unit: one KV head = one rotation chunk = 128 elements.
static constexpr int TURBOQ_HEAD_RECORD_DIM = 128;

// Paper/llama.cpp block size (not used in v1 per-head model).
static constexpr int TURBOQ_PAPER_BLOCK_SIZE = 256;

// Packed sizes per head record (head_dim == 128).
// TBQ4: 128 * 4 bits / 8 = 64 bytes data  +  4 bytes fp32 norm = 68 bytes.
// TBQ3: 128 * 3 bits / 8 = 48 bytes data  +  4 bytes fp32 norm = 52 bytes.
static constexpr size_t TURBOQ_HEAD_BYTES_TBQ4 = 68;
static constexpr size_t TURBOQ_HEAD_BYTES_TBQ3 = 52;

// QJL (Stage 2) constants.
// Sign bits per head record: 128 coordinates = 16 bytes.
static constexpr size_t TURBOQ_SIGN_BYTES = 16;

// QJL head record sizes (head_dim == 128).
// TBQ4+QJL: 48B (3-bit idx) + 16B (signs) + 4B (fp32 norm) + 4B (fp32 gamma) = 72 bytes.
// TBQ3+QJL: 32B (2-bit idx) + 16B (signs) + 4B (fp32 norm) + 4B (fp32 gamma) = 56 bytes.
static constexpr size_t TURBOQ_HEAD_BYTES_TBQ4_QJL = 72;
static constexpr size_t TURBOQ_HEAD_BYTES_TBQ3_QJL = 56;

// Static assertions: codebook symmetry.
static_assert(TURBOQ_CODEBOOK_2BIT[0] == -TURBOQ_CODEBOOK_2BIT[3], "2-bit codebook must be symmetric");
static_assert(TURBOQ_CODEBOOK_2BIT[1] == -TURBOQ_CODEBOOK_2BIT[2], "2-bit codebook must be symmetric");

static_assert(TURBOQ_CODEBOOK_3BIT[0] == -TURBOQ_CODEBOOK_3BIT[7], "3-bit codebook must be symmetric");
static_assert(TURBOQ_CODEBOOK_3BIT[1] == -TURBOQ_CODEBOOK_3BIT[6], "3-bit codebook must be symmetric");
static_assert(TURBOQ_CODEBOOK_3BIT[2] == -TURBOQ_CODEBOOK_3BIT[5], "3-bit codebook must be symmetric");
static_assert(TURBOQ_CODEBOOK_3BIT[3] == -TURBOQ_CODEBOOK_3BIT[4], "3-bit codebook must be symmetric");

static_assert(TURBOQ_CODEBOOK_4BIT[0] == -TURBOQ_CODEBOOK_4BIT[15], "4-bit codebook must be symmetric");
static_assert(TURBOQ_CODEBOOK_4BIT[1] == -TURBOQ_CODEBOOK_4BIT[14], "4-bit codebook must be symmetric");
static_assert(TURBOQ_CODEBOOK_4BIT[7] == -TURBOQ_CODEBOOK_4BIT[8], "4-bit codebook must be symmetric");
