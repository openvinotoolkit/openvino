// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>

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
    -1.51040113F,
    -0.45277205F,
    +0.45277205F,
    +1.51040113F,
};

static constexpr float TURBOQ_CODEBOOK_3BIT[8] = {
    -2.15187458F,
    -1.34383723F,
    -0.75595247F,
    -0.24507484F,
    +0.24507484F,
    +0.75595247F,
    +1.34383723F,
    +2.15187458F,
};

static constexpr float TURBOQ_CODEBOOK_4BIT[16] = {
    -2.73235729F,
    -2.06875481F,
    -1.61777864F,
    -1.25597887F,
    -0.94212216F,
    -0.65659074F,
    -0.38794210F,
    -0.12835875F,
    +0.12835875F,
    +0.38794210F,
    +0.65659074F,
    +0.94212216F,
    +1.25597887F,
    +1.61777864F,
    +2.06875481F,
    +2.73235729F,
};

// ---------------------------------------------------------------------------
// head_dim = 128
// ---------------------------------------------------------------------------
static constexpr float TURBOQ_CODEBOOK_2BIT_D128[4] = {
    -1.50517782F,
    -0.45244580F,
    +0.45244580F,
    +1.50517782F,
};

static constexpr float TURBOQ_CODEBOOK_3BIT_D128[8] = {
    -2.13140531F,
    -1.33653124F,
    -0.75328017F,
    -0.24440650F,
    +0.24440650F,
    +0.75328017F,
    +1.33653124F,
    +2.13140531F,
};

static constexpr float TURBOQ_CODEBOOK_4BIT_D128[16] = {
    -2.68863793F,
    -2.04567026F,
    -1.60407964F,
    -1.24752097F,
    -0.93688023F,
    -0.65345125F,
    -0.38627496F,
    -0.12783692F,
    +0.12783692F,
    +0.38627496F,
    +0.65345125F,
    +0.93688023F,
    +1.24752097F,
    +1.60407964F,
    +2.04567026F,
    +2.68863793F,
};

// ---------------------------------------------------------------------------
// head_dim = 256
// ---------------------------------------------------------------------------
static constexpr float TURBOQ_CODEBOOK_2BIT_D256[4] = {
    -1.50778871F,
    -0.45260987F,
    +0.45260987F,
    +1.50778871F,
};

static constexpr float TURBOQ_CODEBOOK_3BIT_D256[8] = {
    -2.14160037F,
    -1.34017750F,
    -0.75461598F,
    -0.24474095F,
    +0.24474095F,
    +0.75461598F,
    +1.34017750F,
    +2.14160037F,
};

static constexpr float TURBOQ_CODEBOOK_4BIT_D256[16] = {
    -2.71033623F,
    -2.05714788F,
    -1.61089805F,
    -1.25173367F,
    -0.93949232F,
    -0.65501613F,
    -0.38710608F,
    -0.12809709F,
    +0.12809709F,
    +0.38710608F,
    +0.65501613F,
    +0.93949232F,
    +1.25173367F,
    +1.61089805F,
    +2.05714788F,
    +2.71033623F,
};

// ---------------------------------------------------------------------------
// head_dim = 512
// ---------------------------------------------------------------------------
static constexpr float TURBOQ_CODEBOOK_2BIT_D512[4] = {
    -1.50909473F,
    -0.45269120F,
    +0.45269120F,
    +1.50909473F,
};

static constexpr float TURBOQ_CODEBOOK_3BIT_D512[8] = {
    -2.14672756F,
    -1.34200568F,
    -0.75528415F,
    -0.24490797F,
    +0.24490797F,
    +0.75528415F,
    +1.34200568F,
    +2.14672756F,
};

static constexpr float TURBOQ_CODEBOOK_4BIT_D512[16] = {
    -2.72131117F,
    -2.06294084F,
    -1.61433642F,
    -1.25385775F,
    -0.94080983F,
    -0.65580594F,
    -0.38752582F,
    -0.12822853F,
    +0.12822853F,
    +0.38752582F,
    +0.65580594F,
    +0.94080983F,
    +1.25385775F,
    +1.61433642F,
    +2.06294084F,
    +2.72131117F,
};

// ---------------------------------------------------------------------------
// Decision boundaries = midpoints between adjacent centroids (per-dim).
// ---------------------------------------------------------------------------
#define TBQ_MIDS_2BIT(CB) {((CB)[0] + (CB)[1]) / 2.0F, ((CB)[1] + (CB)[2]) / 2.0F, ((CB)[2] + (CB)[3]) / 2.0F}
#define TBQ_MIDS_3BIT(CB)        \
    {((CB)[0] + (CB)[1]) / 2.0F, \
     ((CB)[1] + (CB)[2]) / 2.0F, \
     ((CB)[2] + (CB)[3]) / 2.0F, \
     ((CB)[3] + (CB)[4]) / 2.0F, \
     ((CB)[4] + (CB)[5]) / 2.0F, \
     ((CB)[5] + (CB)[6]) / 2.0F, \
     ((CB)[6] + (CB)[7]) / 2.0F}
#define TBQ_MIDS_4BIT(CB)          \
    {((CB)[0] + (CB)[1]) / 2.0F,   \
     ((CB)[1] + (CB)[2]) / 2.0F,   \
     ((CB)[2] + (CB)[3]) / 2.0F,   \
     ((CB)[3] + (CB)[4]) / 2.0F,   \
     ((CB)[4] + (CB)[5]) / 2.0F,   \
     ((CB)[5] + (CB)[6]) / 2.0F,   \
     ((CB)[6] + (CB)[7]) / 2.0F,   \
     ((CB)[7] + (CB)[8]) / 2.0F,   \
     ((CB)[8] + (CB)[9]) / 2.0F,   \
     ((CB)[9] + (CB)[10]) / 2.0F,  \
     ((CB)[10] + (CB)[11]) / 2.0F, \
     ((CB)[11] + (CB)[12]) / 2.0F, \
     ((CB)[12] + (CB)[13]) / 2.0F, \
     ((CB)[13] + (CB)[14]) / 2.0F, \
     ((CB)[14] + (CB)[15]) / 2.0F}

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
    switch (bits) {
    case 4:
        switch (head_dim) {
        case 128:
            return TURBOQ_CODEBOOK_4BIT_D128;
        case 256:
            return TURBOQ_CODEBOOK_4BIT_D256;
        case 512:
            return TURBOQ_CODEBOOK_4BIT_D512;
        default:
            return TURBOQ_CODEBOOK_4BIT;
        }
    case 3:
        switch (head_dim) {
        case 128:
            return TURBOQ_CODEBOOK_3BIT_D128;
        case 256:
            return TURBOQ_CODEBOOK_3BIT_D256;
        case 512:
            return TURBOQ_CODEBOOK_3BIT_D512;
        default:
            return TURBOQ_CODEBOOK_3BIT;
        }
    case 2:
        switch (head_dim) {
        case 128:
            return TURBOQ_CODEBOOK_2BIT_D128;
        case 256:
            return TURBOQ_CODEBOOK_2BIT_D256;
        case 512:
            return TURBOQ_CODEBOOK_2BIT_D512;
        default:
            return TURBOQ_CODEBOOK_2BIT;
        }
    default:
        throw std::invalid_argument("turboq_codebook: bits must be 2, 3, or 4");
    }
}

inline const float* turboq_boundaries(int bits, int head_dim) {
    switch (bits) {
    case 4:
        switch (head_dim) {
        case 128:
            return TURBOQ_BOUNDARIES_4BIT_D128;
        case 256:
            return TURBOQ_BOUNDARIES_4BIT_D256;
        case 512:
            return TURBOQ_BOUNDARIES_4BIT_D512;
        default:
            return TURBOQ_BOUNDARIES_4BIT;
        }
    case 3:
        switch (head_dim) {
        case 128:
            return TURBOQ_BOUNDARIES_3BIT_D128;
        case 256:
            return TURBOQ_BOUNDARIES_3BIT_D256;
        case 512:
            return TURBOQ_BOUNDARIES_3BIT_D512;
        default:
            return TURBOQ_BOUNDARIES_3BIT;
        }
    case 2:
        switch (head_dim) {
        case 128:
            return TURBOQ_BOUNDARIES_2BIT_D128;
        case 256:
            return TURBOQ_BOUNDARIES_2BIT_D256;
        case 512:
            return TURBOQ_BOUNDARIES_2BIT_D512;
        default:
            return TURBOQ_BOUNDARIES_2BIT;
        }
    default:
        throw std::invalid_argument("turboq_boundaries: bits must be 2, 3, or 4");
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
