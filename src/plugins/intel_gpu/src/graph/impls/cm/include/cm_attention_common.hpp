// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cm/cm.h>
#include <cm/cmtl.h>

//# CM-compiler is C++17
static_assert(__cplusplus >= 201703L);

#define SystolicDepth 8
#define RepeatCount 8
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount
//REG_N
//  Xe1:  8
//  Xe2: 16
#define REG_N (CM_GRF_WIDTH/32)

#define kv_step  REG_K
#define q_step   REG_N

// Q pre-scale math constants:
//   scale_factor = 1 / sqrt(head_size)  (provided by CMFLA_SCALE_FACTOR at JIT time)
//   log2e        = log2(e)              (fixed)
constexpr float scale_factor = CMFLA_SCALE_FACTOR;
constexpr float log2e = 1.4426950408889634f;

// JIT const: chooses the domain the softmax score St = K @ Q^T lives in.
//   1 => Q is pre-multiplied by log2e so the online softmax can call cm_exp (== exp2)
//        directly (Item 13 in OPTIMIZE_PLAN, drops one *log2e off the softmax critical path).
//   0 => Q keeps the natural-log domain and the softmax multiplies by log2e internally.
// Set at JIT time (typically via -DCMFLA_Q_SCALED_BY_LOG2=... from the host Python script);
// defaults to 1. The value MUST match how the kernel actually pre-scales Q -- both the Q
// load (via q_prescale) and the softmax templates read this macro, so keeping them in
// sync is enough to prevent log2e from being applied zero or two times.
#ifndef CMFLA_Q_SCALED_BY_LOG2
#define CMFLA_Q_SCALED_BY_LOG2 1
#endif

// Single Q pre-scale constant used by every SDPA/PA kernel at Q load time.
constexpr float q_prescale = CMFLA_Q_SCALED_BY_LOG2 ? (scale_factor * log2e) : scale_factor;

// JIT const: 1 => use the balanced binary-tree reduction (depth log2(rows)) in the online
//                 softmax; requires rows to be a power of two.
//            0 => use the linear-chain reduction (default).
// Applies uniformly to the SDPA (flashattn) and PA (pageatten) kernel families via the
// `cm_online_softmax_update` dispatch macro defined below.
#ifndef CMFLA_USE_TREE_SOFTMAX
#define CMFLA_USE_TREE_SOFTMAX 0
#endif

static_assert(q_step == 16 || q_step == 8);
static_assert(kv_step == 16);
static_assert(CM_HAS_DPAS);

#define DEBUG_SHOW 0
#if !DEBUG_SHOW
template<typename T, int M, int N>
void show(const matrix<T, M, N> mat, bool isfloat=true) {
}
#else
template<typename T, int M, int N>
void show(const matrix<T, M, N> mat, bool isfloat=true) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            if (isfloat)
                printf("%8.4f,", mat[m][n]);
            else
                printf("%8d,", mat[m][n]);

        }
        printf("],\n");
    }
    printf("]\n");
}
#endif
template <typename T1, typename T2>
CM_INLINE void Transpose_16x16(matrix_ref<T1, 16, 16> in,
                               matrix_ref<T2, 16, 16> out) {
  matrix<T2, 16, 16> bBuf;
  #pragma unroll
  for (int k = 0; k < 16; k++) {
    bBuf.row(k) = in.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
  }
  #pragma unroll
  for (int k = 0; k < 16; k++) {
    out.row(k) = bBuf.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
  }
  for (int k = 0; k < 16; k++) {
    bBuf.row(k) = out.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
  }
  #pragma unroll
  for (int k = 0; k < 16; k++) {
    out.row(k) = bBuf.template select<2, 1, 8, 2>((k * 2) % 16, k / 8);
  }
}

template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
    matrix<T2, 8, 8> temp;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        temp.row(i)     = in.template select<2, 1, 4, 2>(i * 2, 0);
        temp.row(i + 4) = in.template select<2, 1, 4, 2>(i * 2, 1);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out.row(i * 2)     = temp.template select<4, 1, 2, 4>(0, i);
        out.row(i * 2 + 1) = temp.template select<4, 1, 2, 4>(4, i);
    }
}

// function templates cannot be partially specialized; use overloading to achieve the same effect
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
    Transpose_8x8(in, out);
}
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 16, 16> in, matrix_ref<T2, 16, 16> out) {
    Transpose_16x16(in, out);
}
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 16, 8> in, matrix_ref<T2, 8, 16> out) {
    Transpose_8x8(in.select<8, 1, 8, 1>(0,0), out.select<8, 1, 8, 1>(0,0));
    Transpose_8x8(in.select<8, 1, 8, 1>(8,0), out.select<8, 1, 8, 1>(0,8));
}
template <typename T1, typename T2>
inline void Transpose2DMatrix(matrix_ref<T1, 8, 16> in, matrix_ref<T2, 16, 8> out) {
    Transpose_8x8(in.select<8, 1, 8, 1>(0,0), out.select<8, 1, 8, 1>(0,0));
    Transpose_8x8(in.select<8, 1, 8, 1>(0,8), out.select<8, 1, 8, 1>(8,0));
}

template <int n_stride, typename T, int M, int N>
CM_INLINE void slm_read_2d(matrix_ref<T, M, N> out, uint slm, int offset) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_slm_block_read(slm, GENX_DWALIGNED, offset + i*n_stride*sizeof(T), out.row(i));
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_svm_block_read(base + i * pitch, out[i]);
    }
}

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<uint, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        out.row(i).format<uint>() = cm_load<uint, N>(base, offset + i * pitch);
    }
}

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        out.row(i).format<uint>() = cm_load<uint, N/2>(base, offset + i * pitch);
    }
}

template <int M, int N, int num_elem>
CM_INLINE void cm_load_2d_with_tail(matrix_ref<uint, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        auto row_data = out.row(i).format<uint>();
        row_data = 0;
        auto src = cm_load<uint, N>(base, offset + i * pitch);
        row_data.select<num_elem, 1>(0) = src.select<num_elem, 1>(0);
    }
}

template <int M, int N, int num_elem>
CM_INLINE void cm_load_2d_with_tail(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        auto row_data = out.row(i).format<uint>();
        row_data = 0;
        auto src = cm_load<uint, N/2>(base, offset + i * pitch);
        row_data.select<num_elem/2, 1>(0) = src.select<num_elem/2, 1>(0);
    }
}

template <int M, int N>
CM_INLINE void cm_store_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_store<uint, N/2>(base, offset + i * pitch, out.row(i).format<uint>());
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, vector_ref<uint, M> offsets) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_svm_block_read(base + offsets[i], out[i]);
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch, int n_rows) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++, base += pitch, n_rows--) {
        if (n_rows > 0) cm_svm_block_read(base, out[i]);
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_write_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_svm_block_write(base, out[i]);
    }
}

template <typename T, int M, int N>
CM_INLINE void svm_write_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch, int n_rows) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++, base += pitch) {
        if (i < n_rows) cm_svm_block_write(base, out[i]);
    }
}

CM_INLINE uint64_t get_clock() {
    auto clk = cm_clock();
    return ((uint64_t)clk[1]) << 32 | clk[0];
}


template<int num_Qt, int _q_step = REG_N, int _kv_step = REG_K>
inline matrix<float, _kv_step, _q_step> ugemm_KQ(uint slm_K, matrix_ref<half, num_Qt, REG_K*REG_N> Qt, uint slm_offset = 0) {
    matrix<float, _kv_step, _q_step> St;
    constexpr int num_K = _kv_step/REG_M;
    auto St2 = St.format<float, num_K, REG_M*REG_N>();

    matrix<half, num_K, REG_M * REG_K> Kmat;
    cm_slm_block_read(slm_K, GENX_NONE, slm_offset, Kmat.format<half>());
    #pragma unroll
    for(int k = 0; k < num_K; k++)
        St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(0, Qt[0].format<int32_t>(), Kmat[k].format<int32_t>());

    #pragma unroll
    for(int ri = 1; ri < num_Qt; ri++) {
        cm_slm_block_read(slm_K, GENX_NONE, slm_offset + ri * Kmat.n_elems() * sizeof(half), Kmat.format<half>());
        #pragma unroll
        for(int k = 0; k < num_K; k++) {
            St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(St2.row(k), Qt[ri].format<int32_t>(), Kmat[k].format<int32_t>());
        }
    }
    return St;
}

template<int num_P_tiles = REG_N/REG_M, int num_rO_tiles>
inline void ugemm_PV0(uint slm_V, matrix_ref<half, REG_N, REG_K> P, matrix_ref<float, num_rO_tiles, REG_M*REG_N> rO, uint slm_offset = 0) {
    constexpr int _head_size = num_rO_tiles*REG_N/num_P_tiles;

    auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
    #pragma unroll
    for(int k = 0, ri = 0; k < _head_size; k += REG_N, ri += num_P_tiles) {
        matrix<half, REG_K/2, REG_N*2> Vmat;
        cm_slm_block_read(slm_V, GENX_NONE, slm_offset + REG_K*k*sizeof(half), Vmat.format<half>());
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                            0,
                            Vmat.format<int32_t>(),
                            P2.row(p).format<int32_t>());
        }
    }
}

template<int num_P_tiles = REG_N/REG_M, int num_rO_tiles>
inline void ugemm_PV1(uint slm_V, matrix_ref<half, REG_N, REG_K> P, vector_ref<float, REG_N> max_comp,
                      matrix_ref<float, num_rO_tiles, REG_M*REG_N> rO, uint slm_offset = 0) {
    constexpr int _head_size = num_rO_tiles*REG_N/num_P_tiles;
    auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
    #pragma unroll
    for(int k = 0, ri=0; k < _head_size; k += REG_N, ri += num_P_tiles) {
        matrix<half, REG_K/2, REG_N*2> Vmat;

        cm_slm_block_read(slm_V, GENX_NONE, slm_offset + REG_K*k*sizeof(half), Vmat.format<half>());
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < REG_M; r++)
                cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
        }


        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        rO[ri + p].format<float>(),
                        Vmat.format<int32_t>(),
                        P2.row(p).format<int32_t>());
        }
    }
}

// Online-softmax update (linear-chain max/sum reduction).
template<typename T, int rows, int cols>
vector<float, cols> online_softmax_update(matrix_ref<T, rows, cols> St, vector_ref<T, cols> cur_max, vector_ref<T, cols> cur_sum) {
    vector<float, cols> new_max_t;
    new_max_t = cm_max<float>(St[0], St[1]);
    for(int r = 2; r < St.n_rows(); r++) new_max_t = cm_max<float>(new_max_t, St[r]);
    new_max_t = cm_max<float>(new_max_t, cur_max);

    // Pt = torch.exp(St - new_max)
#if CMFLA_Q_SCALED_BY_LOG2
    for(int r = 0; r < St.n_rows(); r++) St[r] = cm_exp(St[r] - new_max_t);
#else
    for(int r = 0; r < St.n_rows(); r++) St[r] = cm_exp((St[r] - new_max_t)*log2e);
#endif

    vector<float, cols> row_sum_t;
    row_sum_t = cm_add<float>(St[0], St[1]);
    for(int r = 2; r < St.n_rows(); r++) row_sum_t = cm_add<float>(row_sum_t, St[r]);

    vector<float, cols> max_comp;
#if CMFLA_Q_SCALED_BY_LOG2
    max_comp = cm_exp(cur_max - new_max_t);
#else
    max_comp = cm_exp((cur_max - new_max_t)*log2e);
#endif
    cur_sum = cm_mul<float>(cur_sum, max_comp);
    cur_sum = cm_add<float>(cur_sum, row_sum_t);
    cur_max = new_max_t;
    return max_comp;
}

// Tree-reduction variant: max/sum reductions use a balanced binary tree (depth log2(rows))
// instead of a linear chain (depth rows-1), shortening the loop-carried dependency chain.
// Requires rows to be a power of two; the PA kv_step=16 and KV_BLK*kv_step satisfy this.
template<typename T, int rows, int cols>
CM_INLINE vector<float, cols> online_softmax_update_tree(matrix_ref<T, rows, cols> St,
                                                         vector_ref<T, cols> cur_max,
                                                         vector_ref<T, cols> cur_sum) {
    static_assert((rows & (rows - 1)) == 0, "tree reduction needs power-of-two rows");
    vector<float, cols> new_max_t;
    {
        matrix<float, (rows > 1 ? rows/2 : 1), cols> t;
        #pragma unroll
        for (int r = 0; r < rows/2; r++) t.row(r) = cm_max<float>(St[r], St[r + rows/2]);
        #pragma unroll
        for (int stride = rows/4; stride > 0; stride >>= 1)
            #pragma unroll
            for (int r = 0; r < stride; r++)
                t.row(r) = cm_max<float>(t.row(r), t.row(r + stride));
        new_max_t = cm_max<float>(t.row(0), cur_max);
    }
#if CMFLA_Q_SCALED_BY_LOG2
    #pragma unroll
    for (int r = 0; r < rows; r++) St[r] = cm_exp(St[r] - new_max_t);
#else
    #pragma unroll
    for (int r = 0; r < rows; r++) St[r] = cm_exp((St[r] - new_max_t) * log2e);
#endif

    vector<float, cols> row_sum_t;
    {
        matrix<float, (rows > 1 ? rows/2 : 1), cols> t;
        #pragma unroll
        for (int r = 0; r < rows/2; r++) t.row(r) = cm_add<float>(St[r], St[r + rows/2]);
        #pragma unroll
        for (int stride = rows/4; stride > 0; stride >>= 1)
            #pragma unroll
            for (int r = 0; r < stride; r++)
                t.row(r) = cm_add<float>(t.row(r), t.row(r + stride));
        row_sum_t = t.row(0);
    }

    vector<float, cols> max_comp;
#if CMFLA_Q_SCALED_BY_LOG2
    max_comp = cm_exp(cur_max - new_max_t);
#else
    max_comp = cm_exp((cur_max - new_max_t) * log2e);
#endif
    cur_sum = cm_mul<float>(cur_sum, max_comp);
    cur_sum = cm_add<float>(cur_sum, row_sum_t);
    cur_max = new_max_t;
    return max_comp;
}

// Dispatch macro used by every SDPA/PA kernel; selects linear vs tree reduction.
#if CMFLA_USE_TREE_SOFTMAX
#define cm_online_softmax_update(St, cur_max, cur_sum) online_softmax_update_tree(St, cur_max, cur_sum)
#else
#define cm_online_softmax_update(St, cur_max, cur_sum) online_softmax_update(St, cur_max, cur_sum)
#endif

// Transpose a float score tile (kv x q) into a half P tile (q x kv) for the P@V matmul.
// Casting float->half first (one vectorized cm_mul-free copy) lets the shuffle network
// run at half width -- roughly halving the mov count vs transposing the float tile
// directly through Transpose_*x*<float,half>.
//
// Two overloads cover both Xe generations (kv_step is always 16; q_step = REG_N):
//   * [16,16] -> [16,16] for Xe2 (q_step = 16)
//   * [16, 8] -> [ 8,16] for Xe1 (q_step =  8)
CM_INLINE void transpose_St_to_P_half(matrix_ref<float, 16, 16> St, matrix_ref<half, 16, 16> P) {
    matrix<half, 16, 16> Sh;
    #pragma unroll
    for (int r = 0; r < 16; r++) Sh.row(r) = St.row(r);
    Transpose_16x16(Sh.select<16,1,16,1>(0,0), P);
}
CM_INLINE void transpose_St_to_P_half(matrix_ref<float, 16, 8> St, matrix_ref<half, 8, 16> P) {
    matrix<half, 16, 8> Sh;
    #pragma unroll
    for (int r = 0; r < 16; r++) Sh.row(r) = St.row(r);
    Transpose2DMatrix(Sh.select<16,1,8,1>(0,0), P);
}

//===============================================================================================
template <int i, int N, int M>
constexpr void apply_causal_mask(matrix_ref<float, N, M> St) {
    if constexpr (i < N) {
        St.row(i).select<i, 1>(0) = -3.4e38f;
        apply_causal_mask<i + 1>(St);
    }
}

template <int N, int M>
inline void apply_causal_mask_with_offset(matrix_ref<float, N, M> St, int causal_left) {
    if (causal_left >= (N - 1)) {
        return;
    }

    #pragma unroll
    for (int r = 0; r < N; r++) {
        int mask_cols = r - causal_left;
        if (mask_cols <= 0) {
            continue;
        }
        if (mask_cols >= M) {
            St.row(r) = -3.4e38f;
            continue;
        }
        #pragma unroll
        for (int c = 0; c < M; c++) {
            if (c < mask_cols) {
                St(r, c) = -3.4e38f;
            }
        }
    }
}

//prepack [K, N] to [K/2, N, 2] layout.
template <typename T1, typename T2, int K, int N>
inline void prepackAsVNNIWidth2(matrix_ref<T1, K, N> input, matrix_ref<T2, K/2, N*2> out) {
    #pragma unroll
    for (int r = 0; r < K/2; r++) {
        out.row(r).select<N, 2>(0) = input.row(r*2);
        out.row(r).select<N, 2>(1) = input.row(r*2+1);
    }
}