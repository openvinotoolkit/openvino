/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/ocl/ocl_math_utils.h"

#if ELEMENT_SIZE == 2
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#define ELEMENT ushort
#define ELEMENT2 ushort2
#define ELEMENT4 ushort4
#define ELEMENT8 ushort8
#define ELEMENT16 ushort16
#define ELEMENT_INT ushort2
#define ELEMENT_INT4 ushort8
#define VLOAD_ELEMENT_INT vload2
#define ELEMENTS_PER_INT 2
#define BLOCK_READ_ELEMENT2 intel_sub_group_block_read_us2
#define BLOCK_READ_ELEMENT4 intel_sub_group_block_read_us4
#define BLOCK_READ_ELEMENT_INT intel_sub_group_block_read_us2
#define MASKED_BLOCK_READ_ELEMENT_INT masked_block_read_element2
#define BLOCK_WRITE_ELEMENT_INT4 intel_sub_group_block_write_us8
#elif ELEMENT_SIZE == 1
#define ELEMENT uchar
#define ELEMENT2 uchar2
#define ELEMENT4 uchar4
#define ELEMENT8 uchar8
#define ELEMENT16 uchar16
#define ELEMENT_INT uchar4
#define ELEMENT_INT4 uchar16
#define VLOAD_ELEMENT_INT vload4
#define BLOCK_READ_ELEMENT2 intel_sub_group_block_read_uc2
#define BLOCK_READ_ELEMENT4 intel_sub_group_block_read_uc4
#define BLOCK_READ_ELEMENT_INT intel_sub_group_block_read_uc4
#define MASKED_BLOCK_READ_ELEMENT_INT masked_block_read_element4
#define BLOCK_WRITE_ELEMENT_INT4 intel_sub_group_block_write_uc16
#define ELEMENTS_PER_INT 4
#define SUM_T int
#define SUM_T4 int4
#define CONVERT_SUM_T convert_int
#define CONVERT_SUM_T4 convert_int4
#if COPY_SIGNED
#define AS_SIGNED_ELEMENT as_char
#define AS_SIGNED_ELEMENT4 as_char4
#define AS_SIGNED_ELEMENT_INT as_char4
#define SIGNED_ELEMENT_INT char4
#else
#define AS_SIGNED_ELEMENT as_uchar
#define AS_SIGNED_ELEMENT4 as_uchar4
#define AS_SIGNED_ELEMENT_INT as_uchar4
#define SIGNED_ELEMENT_INT uchar4
#endif
#else
#error Unsupported element size.
#endif

#if !COPY_A && !COPY_B
#error Source matrix not defined.
#endif

inline ELEMENT2 masked_block_read_element2(global ELEMENT *p, int rem) {
    ELEMENT2 v;
    int lid = get_sub_group_local_id();
    int sg = get_sub_group_size();

    v.s0 = (lid < rem) ? p[lid] : 0;
    v.s1 = (lid + sg < rem) ? p[lid + sg] : 0;

    return v;
}

inline ELEMENT4 masked_block_read_element4(global ELEMENT *p, int rem) {
    ELEMENT4 v;
    int lid = get_sub_group_local_id();
    int sg = get_sub_group_size();

    v.s0 = (lid < rem) ? p[lid] : 0;
    v.s1 = (lid + sg < rem) ? p[lid + sg] : 0;
    v.s2 = (lid + 2 * sg < rem) ? p[lid + 2 * sg] : 0;
    v.s3 = (lid + 3 * sg < rem) ? p[lid + 3 * sg] : 0;

    return v;
}

__attribute__((overloadable)) inline int sum(int v) {
    return sub_group_reduce_add(v);
}

__attribute__((overloadable)) inline int sum(int4 v) {
    return sub_group_reduce_add(v.s0) + sub_group_reduce_add(v.s1)
            + sub_group_reduce_add(v.s2) + sub_group_reduce_add(v.s3);
}

void dummy_dpas() {
    if (get_sub_group_local_id() >= 16) {
        int __builtin_IB_sub_group_idpas_s8_s8_8_1(int, int, int8)
                __attribute__((const));
        global volatile int *_;

        int z = __builtin_IB_sub_group_idpas_s8_s8_8_1(0, _[0], 1);
        for (int i = 0; i < z; i++)
            (void)_[0];
    }
}

#define DUMMY_DPAS dummy_dpas()

#if ELEMENT_SIZE == 2
#define PARTIAL_LOAD(regs, rrem, crem, cc, p) \
    if ((2 * cc + 1) < crem) { \
        if (lid < rrem) regs[cc] = vload2(0, p); \
    } else if ((2 * cc) < crem) { \
        if (lid < rrem) regs[cc].s0 = *(p); \
    }
#elif ELEMENT_SIZE == 1
#define PARTIAL_LOAD(regs, rrem, crem, cc, p) \
    if ((4 * cc + 3) < crem) { \
        if (lid < rrem) regs[cc] = vload4(0, p); \
    } else if ((4 * cc + 2) < crem) { \
        if (lid < rrem) regs[cc].s012 = vload3(0, p); \
    } else if ((4 * cc + 1) < crem) { \
        if (lid < rrem) regs[cc].s01 = vload2(0, p); \
    } else if (4 * cc < crem) { \
        if (lid < rrem) regs[cc].s0 = *(p); \
    }
#endif

#if COPY_A

#define UNROLL_M 32
#define UNROLL_K (32 / ELEMENT_SIZE)

#if COPY_SUM
#define GET_A_SUM_ADDRESS \
    int k_align = (k + UNROLL_K - 1) & ~(UNROLL_K - 1); \
    global int *a_sum = (global int *)(a_packed + offseta_packed \
            + m0 * lda_packed + k_align * UNROLL_M);
#else
#define GET_A_SUM_ADDRESS
#endif

#if COPY_CLEAR_SUM

// A sum clear kernel: initialize row sums to zero.
__attribute__((intel_reqd_sub_group_size(8))) kernel void
xe_hp_systolic_gemm_copy(long m, long k, global ELEMENT *a_packed,
        int offseta_packed, int lda_packed) {

    uint m0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_M;

    GET_A_SUM_ADDRESS;

    uint4 zero = 0;
    intel_sub_group_block_write4(a_sum, zero);
}

#elif !COPY_TRANS

#if ELEMENT_SIZE == 2
#define REPACK_REG(rr, cc) \
    blk_r[rr].s##cc = (((uint)c[2 * cc + 1].s##rr) << 16) | c[2 * cc].s##rr
#elif ELEMENT_SIZE == 1
#define REPACK_REG(rr, cc) \
    blk_r[rr].s##cc = (((uint)c[4 * cc + 3].s##rr) << 24) \
            | (((uint)c[4 * cc + 2].s##rr) << 16) \
            | (((uint)c[4 * cc + 1].s##rr) << 8) | c[4 * cc].s##rr
#endif

#define REPACK_CC(cc) \
    REPACK_REG(0, cc); \
    REPACK_REG(1, cc); \
    REPACK_REG(2, cc); \
    REPACK_REG(3, cc)

#define REPACK \
    REPACK_CC(0); \
    REPACK_CC(1); \
    REPACK_CC(2); \
    REPACK_CC(3); \
    REPACK_CC(4); \
    REPACK_CC(5); \
    REPACK_CC(6); \
    REPACK_CC(7)

// Nontranspose A copy.
// Each thread packs a 32x16 (f16/bf16) or 32x32 (u8/s8) block of A.
__attribute__((intel_reqd_sub_group_size(8))) kernel void
xe_hp_systolic_gemm_copy(long m, long k, global ELEMENT *a, long offseta,
        long lda, global ELEMENT *a_packed, int offseta_packed,
        int lda_packed) {

    int lid = get_sub_group_local_id();
    uint m0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_M;
    uint k0 = get_global_id(1) * UNROLL_K;
    int mrem = m - m0;
    int krem = k - k0;
    bool aligned = ((as_long(a) | lda | offseta) & (ELEMENTS_PER_INT - 1)) == 0;

    if (mrem <= 0 || krem <= 0) return;

    GET_A_SUM_ADDRESS;

    a += offseta + m0 + k0 * lda;
    a_packed += offseta_packed + m0 * lda_packed + k0 * UNROLL_M;

    // Read all columns.
    ELEMENT4 c[UNROLL_K];

    if (mrem >= UNROLL_M && krem >= UNROLL_K && aligned) {
        for (int h = 0; h < UNROLL_K; h++)
            c[h] = BLOCK_READ_ELEMENT4(a + h * lda);
    } else {
        for (int h = 0; h < UNROLL_K; h++)
            if (h < krem)
                c[h] = masked_block_read_element4(a + h * lda, mrem);
            else
                c[h] = 0;
    }

    // Rearrange.
    uint8 blk_r[UNROLL_M / 8];
    REPACK;

    // Write out.
    for (int rr = 0; rr < UNROLL_M / 8; rr++)
        intel_sub_group_block_write8(
                (global uint *)(a_packed + rr * UNROLL_K * 8), blk_r[rr]);

        // Sum if needed.
#if COPY_SUM
    SUM_T4 sum = 0;
    for (int h = 0; h < UNROLL_K; h++)
        sum += CONVERT_SUM_T4(AS_SIGNED_ELEMENT4(c[h]));
    atomic_add(a_sum + lid, sum.s0);
    atomic_add(a_sum + lid + 8, sum.s1);
    atomic_add(a_sum + lid + 16, sum.s2);
    atomic_add(a_sum + lid + 24, sum.s3);
#endif

    DUMMY_DPAS;
}

#else /* COPY_TRANS */

// Transpose A copy.
__attribute__((intel_reqd_workgroup_walk_order(1, 0)))
__attribute__((intel_reqd_sub_group_size(8))) kernel void
xe_hp_systolic_gemm_copy(long m, long k, global ELEMENT *a, long offseta,
        long lda, global ELEMENT *a_packed, int offseta_packed,
        int lda_packed) {

    int lid = get_sub_group_local_id();
    uint m0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_M;
    uint k0 = get_global_id(1) * UNROLL_K;
    int mrem = m - m0;
    int krem = k - k0;

    if (mrem <= 0 || krem <= 0) return;

    GET_A_SUM_ADDRESS;

    a += offseta + m0 * lda + k0;
    a_packed += offseta_packed + m0 * lda_packed + k0 * UNROLL_M;

#if COPY_SUM
    SUM_T sum[UNROLL_M / 8] = {0};
#endif

    for (int rr = 0; rr < UNROLL_M / 8; rr++, mrem -= 8) {
        ELEMENT_INT regs[8];

        if (mrem >= UNROLL_M && krem >= UNROLL_K) {
            for (int cc = 0; cc < UNROLL_K / ELEMENTS_PER_INT; cc++)
                regs[cc] = VLOAD_ELEMENT_INT(0,
                        a + ((rr * 8) + lid) * lda + (cc * ELEMENTS_PER_INT));
        } else {
            for (int cc = 0; cc < UNROLL_K / ELEMENTS_PER_INT; cc++) {
                regs[cc] = 0;
                PARTIAL_LOAD(regs, mrem, krem, cc,
                        a + ((rr * 8) + lid) * lda + (cc * ELEMENTS_PER_INT));
            }
        }

        uint8 blk_r;
        blk_r.s0 = as_uint(regs[0]);
        blk_r.s1 = as_uint(regs[1]);
        blk_r.s2 = as_uint(regs[2]);
        blk_r.s3 = as_uint(regs[3]);
        blk_r.s4 = as_uint(regs[4]);
        blk_r.s5 = as_uint(regs[5]);
        blk_r.s6 = as_uint(regs[6]);
        blk_r.s7 = as_uint(regs[7]);

#if COPY_SUM
        for (int cc = 0; cc < UNROLL_K / ELEMENTS_PER_INT; cc++) {
            sum[rr] += CONVERT_SUM_T(AS_SIGNED_ELEMENT(regs[cc].s0));
            sum[rr] += CONVERT_SUM_T(AS_SIGNED_ELEMENT(regs[cc].s1));
            sum[rr] += CONVERT_SUM_T(AS_SIGNED_ELEMENT(regs[cc].s2));
            sum[rr] += CONVERT_SUM_T(AS_SIGNED_ELEMENT(regs[cc].s3));
        }
#endif

        intel_sub_group_block_write8(
                (global uint *)(a_packed + rr * UNROLL_K * 8), blk_r);
    }

#if COPY_SUM
    atomic_add(a_sum + lid, sum[0]);
    atomic_add(a_sum + lid + 8, sum[1]);
    atomic_add(a_sum + lid + 16, sum[2]);
    atomic_add(a_sum + lid + 24, sum[3]);
#endif

    DUMMY_DPAS;
}

#endif /* !COPY_TRANS */
#endif /* COPY_A */

#if COPY_B

#define UNROLL_K (32 / ELEMENT_SIZE)

#if ELEMENT_SIZE == 2
#define REPACK_CC(cc) \
    do { \
        colgroups[cc].s01 = cols[cc * 4]; \
        colgroups[cc].s23 = cols[cc * 4 + 1]; \
        colgroups[cc].s45 = cols[cc * 4 + 2]; \
        colgroups[cc].s67 = cols[cc * 4 + 3]; \
    } while (false)
#define REPACK_CC2(cc) \
    do { \
        colgroups[cc].s02 = cols[cc * 2]; \
        colgroups[cc].s13 = cols2[cc * 2]; \
        colgroups[cc].s46 = cols[cc * 2 + 1]; \
        colgroups[cc].s57 = cols2[cc * 2 + 1]; \
    } while (false)
#elif ELEMENT_SIZE == 1
#define REPACK_CC(cc) \
    do { \
        colgroups[cc].s0123 = cols[cc * 4]; \
        colgroups[cc].s4567 = cols[cc * 4 + 1]; \
        colgroups[cc].s89ab = cols[cc * 4 + 2]; \
        colgroups[cc].scdef = cols[cc * 4 + 3]; \
    } while (false)
#define REPACK_CC4(cc) \
    do { \
        colgroups[cc].s048c = cols[cc]; \
        colgroups[cc].s159d = cols2[cc]; \
        colgroups[cc].s26ae = cols3[cc]; \
        colgroups[cc].s37bf = cols4[cc]; \
    } while (false)
#endif

#if COPY_SUM
#define GET_B_SUM_ADDRESS \
    int k_align = (k + UNROLL_K - 1) & ~(UNROLL_K - 1); \
    global int *b_sum = (global int *)(b_packed + offsetb_packed \
            + n0 * ldb_packed + k_align * UNROLL_N);
#else
#define GET_B_SUM_ADDRESS
#endif

#if COPY_CLEAR_SUM

// B sum clear kernel: initialize column sums to zero.
__attribute__((intel_reqd_sub_group_size(8))) kernel void
xe_hp_systolic_gemm_copy(long k, long n, global ELEMENT *b_packed,
        int offsetb_packed, int ldb_packed) {

    uint n0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_N;

    GET_B_SUM_ADDRESS;

    uint4 zero = 0;
    intel_sub_group_block_write4(b_sum, zero);
#if UNROLL_N > 32
    intel_sub_group_block_write2(b_sum + 32, zero.s01);
#endif
}

#elif !COPY_TRANS

// Each thread packs a 16x{32,48} (f16/bf16) or 32x{32,48} (u8/s8) block of B.
// Nontranspose B copy.
__attribute__((intel_reqd_sub_group_size(8))) kernel void
xe_hp_systolic_gemm_copy(long k, long n, global ELEMENT *b, long offsetb,
        long ldb, global ELEMENT *b_packed, int offsetb_packed,
        int ldb_packed) {

    int lid = get_sub_group_local_id();
    uint k0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_K;
    uint n0 = get_global_id(1) * UNROLL_N;
    int krem = k - k0;
    int nrem = n - n0;
    bool aligned = ((as_long(b) | ldb | offsetb) & (ELEMENTS_PER_INT - 1)) == 0;

    if (nrem <= 0 || krem <= 0) return;

    GET_B_SUM_ADDRESS;
    b += offsetb + k0 + n0 * ldb;
    b_packed += offsetb_packed + n0 * ldb_packed + k0 * UNROLL_N;

    // Copy in two halves.

#define UNROLL_N_CHUNK (UNROLL_N / 2)
#if COPY_SUM
    SUM_T sums[UNROLL_N];
#endif
    ELEMENT_INT cols[UNROLL_N / 2];

    for (int c0 = 0; c0 < UNROLL_N;
            c0 += UNROLL_N_CHUNK, nrem -= UNROLL_N_CHUNK) {
        // Read all columns.
        if (krem >= UNROLL_K && nrem >= UNROLL_N_CHUNK && aligned) {
            for (int c = 0; c < UNROLL_N_CHUNK; c++)
                cols[c] = BLOCK_READ_ELEMENT_INT(b + (c + c0) * ldb);
        } else {
            for (int c = 0; c < UNROLL_N_CHUNK; c++)
                if (c < nrem)
                    cols[c] = MASKED_BLOCK_READ_ELEMENT_INT(
                            b + (c + c0) * ldb, krem);
                else
                    cols[c] = 0;
        }

        // Repack.
        ELEMENT_INT4 colgroups[UNROLL_N_CHUNK / 4];
        for (int cc = 0; cc < UNROLL_N_CHUNK / 4; cc++)
            REPACK_CC(cc);

        // Write out.
        for (int cc = 0; cc < UNROLL_N_CHUNK / 4; cc++)
            BLOCK_WRITE_ELEMENT_INT4(
                    b_packed + (cc * 4 + c0) * UNROLL_K, colgroups[cc]);

            // Sum if needed.
#if COPY_SUM
        for (int c = 0; c < UNROLL_N_CHUNK; c++)
            sums[c + c0] = sum(CONVERT_SUM_T4(AS_SIGNED_ELEMENT_INT(cols[c])));
#endif
    }

    // Accumulate sums.
#if COPY_SUM
    for (int c0 = 0; c0 < UNROLL_N; c0 += get_sub_group_size())
        atomic_add(b_sum + c0 + lid, sums[c0 + lid]);
#endif

    DUMMY_DPAS;
}

#else /* COPY_TRANS */

#define ADD_SUM(coln) \
    for (int cc = 0; cc < UNROLL_N / 4; cc++) { \
        sums[4 * cc + 0] \
                += sum(CONVERT_SUM_T(AS_SIGNED_ELEMENT(coln[cc].s0))); \
        sums[4 * cc + 1] \
                += sum(CONVERT_SUM_T(AS_SIGNED_ELEMENT(coln[cc].s1))); \
        sums[4 * cc + 2] \
                += sum(CONVERT_SUM_T(AS_SIGNED_ELEMENT(coln[cc].s2))); \
        sums[4 * cc + 3] \
                += sum(CONVERT_SUM_T(AS_SIGNED_ELEMENT(coln[cc].s3))); \
    }

// Transpose B copy.
__attribute__((intel_reqd_workgroup_walk_order(1, 0)))
__attribute__((intel_reqd_sub_group_size(8))) kernel void
xe_hp_systolic_gemm_copy(long k, long n, global ELEMENT *b, long offsetb,
        long ldb, global ELEMENT *b_packed, int offsetb_packed,
        int ldb_packed) {

    int lid = get_sub_group_local_id();
    uint k0 = (sub_group_broadcast(get_global_id(0), 0) / 8) * UNROLL_K;
    uint n0 = get_global_id(1) * UNROLL_N;
    int krem = k - k0;
    int nrem = n - n0;
    int sg = get_sub_group_size();

    if (nrem <= 0 || krem <= 0) return;

    GET_B_SUM_ADDRESS;
    b += offsetb + n0 + k0 * ldb;
    b_packed += offsetb_packed + n0 * ldb_packed + k0 * UNROLL_N;

    // Read upper 16x{32,48} submatrix.
    ELEMENT_INT cols[UNROLL_N / ELEMENTS_PER_INT];
    ELEMENT_INT cols2[UNROLL_N / ELEMENTS_PER_INT];
    ELEMENT_INT4 colgroups[UNROLL_N / 4];
    if (krem >= 2 * sg && nrem >= UNROLL_N) {
        for (int cc = 0; cc < UNROLL_N / ELEMENTS_PER_INT; cc++) {
            cols[cc] = VLOAD_ELEMENT_INT(
                    0, b + cc * ELEMENTS_PER_INT + lid * ldb);
            cols2[cc] = VLOAD_ELEMENT_INT(
                    0, b + cc * ELEMENTS_PER_INT + (lid + sg) * ldb);
        }
    } else {
        for (int cc = 0; cc < UNROLL_N / ELEMENTS_PER_INT; cc++) {
            cols[cc] = 0;
            cols2[cc] = 0;
            PARTIAL_LOAD(cols, krem, nrem, cc,
                    b + cc * ELEMENTS_PER_INT + lid * ldb);
            PARTIAL_LOAD(cols2, krem - sg, nrem, cc,
                    b + cc * ELEMENTS_PER_INT + (lid + sg) * ldb);
        }
    }
#if ELEMENT_SIZE == 2
    // Repack.
    for (int cc = 0; cc < UNROLL_N / 4; cc++)
        REPACK_CC2(cc);
#else
    // Read lower 16x{32,48} submatrix.
    ELEMENT_INT cols3[UNROLL_N / ELEMENTS_PER_INT];
    ELEMENT_INT cols4[UNROLL_N / ELEMENTS_PER_INT];
    krem -= 2 * sg;
    if (krem >= 2 * sg && nrem >= UNROLL_N) {
        for (int cc = 0; cc < UNROLL_N / ELEMENTS_PER_INT; cc++) {
            cols3[cc] = VLOAD_ELEMENT_INT(
                    0, b + cc * ELEMENTS_PER_INT + (lid + 2 * sg) * ldb);
            cols4[cc] = VLOAD_ELEMENT_INT(
                    0, b + cc * ELEMENTS_PER_INT + (lid + 3 * sg) * ldb);
        }
    } else {
        for (int cc = 0; cc < UNROLL_N / ELEMENTS_PER_INT; cc++) {
            cols3[cc] = 0;
            cols4[cc] = 0;
            PARTIAL_LOAD(cols3, krem, nrem, cc,
                    b + cc * ELEMENTS_PER_INT + (lid + 2 * sg) * ldb);
            PARTIAL_LOAD(cols4, krem - sg, nrem, cc,
                    b + cc * ELEMENTS_PER_INT + (lid + 3 * sg) * ldb);
        }
    }
    for (int cc = 0; cc < UNROLL_N / 4; cc++)
        REPACK_CC4(cc);
#endif

    // Write out.
    for (int cc = 0; cc < UNROLL_N / 4; cc++)
        BLOCK_WRITE_ELEMENT_INT4(b_packed + cc * 4 * UNROLL_K, colgroups[cc]);

#if COPY_SUM
    SUM_T sums[UNROLL_N] = {0};
    ADD_SUM(cols);
    ADD_SUM(cols2);
    ADD_SUM(cols3);
    ADD_SUM(cols4);

    for (int c0 = 0; c0 < UNROLL_N; c0 += get_sub_group_size())
        atomic_add(b_sum + c0 + lid, sums[c0 + lid]);
#endif

    DUMMY_DPAS;
}

#endif /* !COPY_TRANS */
#endif /* COPY_B */
