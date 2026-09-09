/*******************************************************************************
 * Copyright (c) 2018-2026 Intel Corporation
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

// Systolic (cm_dpas) tile geometry and operand builders shared by the QSA prefill kernels.
//
// dpas contract (matches cm_attention_common.hpp::ugemm_KQ):
//   cm_dpas<HF, HF, SystolicDepth, RepeatCount, float>(acc, src1, src2)
//     src1 = B operand, QSA_REG_K x QSA_REG_N halfs in VNNI order [k/2][n][2]
//     src2 = A operand, RepeatCount x QSA_REG_K halfs, row major
//     dst  = RepeatCount x QSA_REG_N floats, dst[m][n] = sum_k A[m][k] * B[k][n]
//
// RepeatCount is NOT required to be 8 - it may be 1, 2, 4 or 8 (pa_small_q.cm relies on
// this). The N dimension is fixed to the GRF width, so it is the axis that must be filled
// with 16 independent outputs; in prefill that axis is the query tile.

// The tile-geometry macros must sit OUTSIDE the include guard. The codegen appends an
// `#undef` for every bodied `#define` at the end of each kernel it inlines this header
// into, while the guard suppresses the second textual copy - so a guarded definition
// would leave them undefined for every kernel after the first one in a batched program.
// (Same trap as ATTR in qsa_common.hpp.) They are plain constants, so redefining them
// per kernel is harmless; `#ifndef QSA_REG_K` keeps it idempotent within one kernel.
#ifndef QSA_REG_K
#define QSA_SYSTOLIC_DEPTH 8
#define QSA_VNNI_WIDTH     2
#define QSA_REG_K          (QSA_SYSTOLIC_DEPTH * QSA_VNNI_WIDTH)  // 16 fp16 K elements per dpas
#define QSA_REG_M          8                                      // max RepeatCount
#define QSA_REG_N          (CM_GRF_WIDTH / 32)                    // 8 on Xe1, 16 on Xe2
#endif

#ifndef QSA_DPAS_COMMON_HPP
#define QSA_DPAS_COMMON_HPP

#include "qsa_common.hpp"

namespace qsa_dpas {

// 8x8 register transpose (same butterfly as cm_attention_common.hpp::Transpose_8x8).
template <typename T1, typename T2>
CM_INLINE void transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
    matrix<T2, 8, 8> temp;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        temp.row(i) = in.template select<2, 1, 4, 2>(i * 2, 0);
        temp.row(i + 4) = in.template select<2, 1, 4, 2>(i * 2, 1);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out.row(i * 2) = temp.template select<4, 1, 2, 4>(0, i);
        out.row(i * 2 + 1) = temp.template select<4, 1, 2, 4>(4, i);
    }
}

/// [8 x 16] -> [16 x 8]; turns a dpas result tile (rows = blocks/KV, cols = queries)
/// back into the query-major layout the metadata buffers use.
template <typename T1, typename T2>
CM_INLINE void transpose_8x16(matrix_ref<T1, 8, 16> in, matrix_ref<T2, 16, 8> out) {
    transpose_8x8(in.template select<8, 1, 8, 1>(0, 0), out.template select<8, 1, 8, 1>(0, 0));
    transpose_8x8(in.template select<8, 1, 8, 1>(0, 8), out.template select<8, 1, 8, 1>(8, 0));
}

/// 16x16 register transpose (same butterfly as cm_attention_common.hpp::Transpose_16x16).
/// Used to turn the softmax result St[kv][query] into the PV A operand P^T[query][kv].
template <typename T1, typename T2>
CM_INLINE void transpose_16x16(matrix_ref<T1, 16, 16> in, matrix_ref<T2, 16, 16> out) {
    matrix<T2, 16, 16> buf;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        buf.row(k) = in.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
        out.row(k) = buf.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
        buf.row(k) = out.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
    }
#pragma unroll
    for (int k = 0; k < 16; k++) {
        out.row(k) = buf.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
    }
}

#ifdef CM_HAS_LSC_UNTYPED_2D

/// Builds one VNNI B operand tile straight from row-major [rows][K] memory.
///
/// A transposed d32 load is exactly a VNNI re-layout for fp16: each loaded uint holds the
/// half pair (k, k+1) of one row, and the transpose puts those pairs at [k/2][row][2] -
/// which is the B operand layout. `rows` is clamped by the descriptor's surface height, so
/// out-of-range rows come back as zeros and need no explicit masking.
///
/// dst holds QSA_REG_K * QSA_REG_N halfs.
CM_INLINE void load_vnni_tile(vector_ref<half, QSA_REG_K * QSA_REG_N> dst,
                              const half* base ATTR,
                              uint rows,
                              uint row_pitch_bytes,
                              uint width_bytes,
                              uint k_offset) {
    lsc::block_2d_desc<uint, 1, QSA_REG_N, QSA_REG_K / 2> desc(reinterpret_cast<uint*>(const_cast<half*>(base)),
                                                               rows - 1,
                                                               width_bytes - 1,
                                                               row_pitch_bytes - 1,
                                                               0,
                                                               0);
    cm_load<lsc::Transpose>(dst.format<uint>(), desc.set_block_x(k_offset / 2));
}

/// Same transposed load, but filling only `COLS` of the B operand's N columns.
///
/// The transpose turns memory ROWS into operand COLUMNS, so a tile whose 16 columns are
/// not 16 consecutive rows of one matrix has to be assembled from several of these. That
/// is what the head-tiled sparse attention needs: its columns are (head, token) pairs, and
/// only the tokens of one head are consecutive rows.
template <int COLS>
CM_INLINE void load_vnni_cols(matrix_ref<uint, QSA_REG_K / 2, COLS> dst,
                              const half* base ATTR,
                              uint rows,
                              uint row_pitch_bytes,
                              uint width_bytes,
                              uint k_offset) {
    lsc::block_2d_desc<uint, 1, COLS, QSA_REG_K / 2> desc(reinterpret_cast<uint*>(const_cast<half*>(base)),
                                                          rows - 1,
                                                          width_bytes - 1,
                                                          row_pitch_bytes - 1,
                                                          0,
                                                          0);
    cm_load<lsc::Transpose>(dst.template format<uint>(), desc.set_block_x(k_offset / 2));
}

/// Loads an A operand tile: `ROWS` consecutive rows of `QSA_REG_K` halfs from row-major memory.
template <int ROWS>
CM_INLINE void load_a_tile(vector_ref<half, ROWS * QSA_REG_K> dst,
                           const half* base ATTR,
                           uint rows,
                           uint row_pitch_bytes,
                           uint width_bytes,
                           uint k_offset) {
    lsc::block_2d_desc<half, 1, ROWS, QSA_REG_K> desc(const_cast<half*>(base), rows - 1, width_bytes - 1, row_pitch_bytes - 1, 0, 0);
    cm_load<lsc::Normal>(dst, desc.set_block_x(k_offset));
}

/// Loads `ROWS` rows x `COLS` halfs in ONE message.
///
/// A QSA block is only `r` (4) rows tall, so a per-dpas-tile load moves just
/// ROWS * QSA_REG_K * 2 = 128 bytes - a quarter of what the 16-row tiles in
/// cm_pa_xe2.hpp / cm_sdpa_common.hpp move, and the sparse attention kernel turned out to
/// be bound by message rate rather than bandwidth. The rows cannot grow (the selected
/// blocks are scattered), but the columns can: one block's `r` tokens are contiguous over
/// the whole head dimension in the paged cache, so the worker's entire slice can come in a
/// single message and be split into dpas tiles in registers.
template <int ROWS, int COLS>
CM_INLINE void load_a_wide(matrix_ref<half, ROWS, COLS> dst,
                           const half* base ATTR,
                           uint rows,
                           uint row_pitch_bytes,
                           uint width_bytes,
                           uint col_offset) {
    lsc::block_2d_desc<half, 1, ROWS, COLS> desc(const_cast<half*>(base), rows - 1, width_bytes - 1, row_pitch_bytes - 1, 0, 0);
    cm_load<lsc::Normal>(dst.template format<half>(), desc.set_block_x(col_offset));
}

/// Same widening for the PV B operand. The hardware VNNI transform yields
/// [ROWS/2][COLS * 2], in which the dpas tile for output dims [d*N, (d+1)*N) is the
/// contiguous column range [d*2N, (d+1)*2N).
template <int ROWS, int COLS>
CM_INLINE void load_vnni_wide(matrix_ref<half, ROWS / 2, COLS * 2> dst,
                              const half* base ATTR,
                              uint rows,
                              uint row_pitch_bytes,
                              uint width_bytes,
                              uint col_offset) {
    static_assert(ROWS % 2 == 0, "VNNI row pairs require an even row count");
    lsc::block_2d_desc<half, 1, ROWS, COLS> desc(const_cast<half*>(base), rows - 1, width_bytes - 1, row_pitch_bytes - 1, 0, 0);
    cm_load<lsc::VNNI>(dst.template format<half>(), desc.set_block_x(col_offset));
}

/// Loads `ROWS` rows x QSA_REG_N columns and lets the hardware apply the VNNI re-layout,
/// producing [ROWS/2][QSA_REG_N * 2] halfs - the B operand slice for a PV dpas.
template <int ROWS>
CM_INLINE void load_vnni_rows(vector_ref<half, ROWS * QSA_REG_N> dst,
                              const half* base ATTR,
                              uint rows,
                              uint row_pitch_bytes,
                              uint width_bytes,
                              uint n_offset) {
    static_assert(ROWS % 2 == 0, "VNNI row pairs require an even row count");
    lsc::block_2d_desc<half, 1, ROWS, QSA_REG_N> desc(const_cast<half*>(base), rows - 1, width_bytes - 1, row_pitch_bytes - 1, 0, 0);
    cm_load<lsc::VNNI>(dst, desc.set_block_x(n_offset));
}

#endif  // CM_HAS_LSC_UNTYPED_2D

}  // namespace qsa_dpas

#endif  // QSA_DPAS_COMMON_HPP
