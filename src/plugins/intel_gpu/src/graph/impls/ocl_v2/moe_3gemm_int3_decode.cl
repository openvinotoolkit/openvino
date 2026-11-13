// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Decoding of the u3 MoE expert weights, shared by the batched GEMV (moe_3gemm_int3.cl) and the int8 GEMMs
// (moe_3gemm_int3_gemm.cl). Weights are the plain [E, N, K] u3 tensor: each output channel is a LSB-first bit stream
// of K values, so a run of 32 values (a granule) is exactly 3 uints.

#define K_CHUNK       32  // u3 values per granule
#define GRANULE_UINTS 3

// Eight consecutive u3 values (the low 24 bits of f; higher bits are ignored)
// spread into the eight nibbles of an int: into 12-bit halves at bits 0 and 16,
// then 6-bit quarters at each byte, then 3-bit values at each nibble. Each step
// keeps the low part in place and takes the rest from the shifted copy (one bfn);
// the bits the steps leave over are cleared at the end.
inline int FUNC(u3_spread8_nib)(uint f) {
    uint u = bitselect(f << 4, f, 0x00000FFFu);
    u = bitselect(u << 2, u, 0x003F003Fu);
    u = bitselect(u << 1, u, 0x07070707u);
    return as_int(u & 0x77777777u);
}
