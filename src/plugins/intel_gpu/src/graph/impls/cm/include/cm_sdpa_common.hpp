/*******************************************************************************
 * Copyright (c) 2022-2025 Intel Corporation
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

constexpr float scale_factor = CMFLA_SCALE_FACTOR;

static_assert(q_step == 16 || q_step == 8);
static_assert(kv_step == 16);
static_assert(CM_HAS_DPAS);

template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template <typename T1, typename T2>
CM_INLINE void Transpose_16x16(matrix_ref<T1, 16, 16> in,
                               matrix_ref<T2, 16, 16> out) {
  matrix<T2, 16, 16> bBuf;
  bBuf.row(0) = in.template select<4, 1, 4, 4>(0, 0);   // 0,4,8,c
  bBuf.row(1) = in.template select<4, 1, 4, 4>(4, 0);   // 0,4,8,c
  bBuf.row(2) = in.template select<4, 1, 4, 4>(8, 0);   // 0,4,8,c
  bBuf.row(3) = in.template select<4, 1, 4, 4>(12, 0);  // 0,4,8,c
  bBuf.row(4) = in.template select<4, 1, 4, 4>(0, 1);   // 1,5,9,d
  bBuf.row(5) = in.template select<4, 1, 4, 4>(4, 1);   // 1,5,9,d
  bBuf.row(6) = in.template select<4, 1, 4, 4>(8, 1);   // 1,5,9,d
  bBuf.row(7) = in.template select<4, 1, 4, 4>(12, 1);  // 1,5,9,d
  bBuf.row(8) = in.template select<4, 1, 4, 4>(0, 2);   // 2,6,a,e
  bBuf.row(9) = in.template select<4, 1, 4, 4>(4, 2);   // 2,6,a,e
  bBuf.row(10) = in.template select<4, 1, 4, 4>(8, 2);  // 2,6,a,e
  bBuf.row(11) = in.template select<4, 1, 4, 4>(12, 2); // 2,6,a,e
  bBuf.row(12) = in.template select<4, 1, 4, 4>(0, 3);  // 3,7,b,f
  bBuf.row(13) = in.template select<4, 1, 4, 4>(4, 3);  // 3,7,b,f
  bBuf.row(14) = in.template select<4, 1, 4, 4>(8, 3);  // 3,7,b,f
  bBuf.row(15) = in.template select<4, 1, 4, 4>(12, 3); // 3,7,b,f

  out.row(0) = bBuf.template select<4, 1, 4, 4>(0, 0);   // 0
  out.row(1) = bBuf.template select<4, 1, 4, 4>(4, 0);   // 1
  out.row(2) = bBuf.template select<4, 1, 4, 4>(8, 0);   // 2
  out.row(3) = bBuf.template select<4, 1, 4, 4>(12, 0);  // 3
  out.row(4) = bBuf.template select<4, 1, 4, 4>(0, 1);   // 4
  out.row(5) = bBuf.template select<4, 1, 4, 4>(4, 1);   // 5
  out.row(6) = bBuf.template select<4, 1, 4, 4>(8, 1);   // 6
  out.row(7) = bBuf.template select<4, 1, 4, 4>(12, 1);  // 7
  out.row(8) = bBuf.template select<4, 1, 4, 4>(0, 2);   // 8
  out.row(9) = bBuf.template select<4, 1, 4, 4>(4, 2);   // 9
  out.row(10) = bBuf.template select<4, 1, 4, 4>(8, 2);  // a
  out.row(11) = bBuf.template select<4, 1, 4, 4>(12, 2); // b
  out.row(12) = bBuf.template select<4, 1, 4, 4>(0, 3);  // c
  out.row(13) = bBuf.template select<4, 1, 4, 4>(4, 3);  // d
  out.row(14) = bBuf.template select<4, 1, 4, 4>(8, 3);  // e
  out.row(15) = bBuf.template select<4, 1, 4, 4>(12, 3); // f
}

template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
  matrix<T2, 8, 8> temp;
  temp.row(0) = in.template select<2, 1, 4, 2>(0, 0);
  temp.row(1) = in.template select<2, 1, 4, 2>(2, 0);
  temp.row(2) = in.template select<2, 1, 4, 2>(4, 0);
  temp.row(3) = in.template select<2, 1, 4, 2>(6, 0);
  temp.row(4) = in.template select<2, 1, 4, 2>(0, 1);
  temp.row(5) = in.template select<2, 1, 4, 2>(2, 1);
  temp.row(6) = in.template select<2, 1, 4, 2>(4, 1);
  temp.row(7) = in.template select<2, 1, 4, 2>(6, 1);

  out.row(0) = temp.template select<4, 1, 2, 4>(0, 0);
  out.row(2) = temp.template select<4, 1, 2, 4>(0, 1);
  out.row(4) = temp.template select<4, 1, 2, 4>(0, 2);
  out.row(6) = temp.template select<4, 1, 2, 4>(0, 3);
  out.row(1) = temp.template select<4, 1, 2, 4>(4, 0);
  out.row(3) = temp.template select<4, 1, 2, 4>(4, 1);
  out.row(5) = temp.template select<4, 1, 2, 4>(4, 2);
  out.row(7) = temp.template select<4, 1, 2, 4>(4, 3);
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
            // if (cm_local_id(1) == 0 && cm_group_id(0) == 0) {
            //     printf("Vmat\n");
            //     show(Vmat);
            //     // printf("P2\n");
            //     // show(P2.row(p).format<half, REG_M, REG_N>());
            // }
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

        //# compensate cur_O
        //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < REG_M; r++)
                cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
        }

        //show(rO[ri].format<float, REG_M, REG_N>());

        //# show(cur_O.format<float, 2*REG_M, REG_N>()); return;
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                        rO[ri + p].format<float>(),
                        Vmat.format<int32_t>(),
                        P2.row(p).format<int32_t>());
            //if (kv_pos == args_verbose) show(rO[ri + p].format<float, REG_M, REG_N>());
        }
        // if (kv_pos == args_verbose) show(cur_O.format<float, 2*REG_M, REG_N>());
    }
}

template<typename T, int rows, int cols>
vector<float, cols> online_softmax_update(matrix_ref<T, rows, cols> St, vector_ref<T, cols> cur_max, vector_ref<T, cols> cur_sum) {
    vector<float, cols> new_max_t;
    new_max_t = cm_max<float>(St[0], St[1]);
    for(int r = 2; r < St.n_rows(); r++) new_max_t = cm_max<float>(new_max_t, St[r]);
    new_max_t = cm_max<float>(new_max_t, cur_max);

    // Pt = torch.exp(St - new_max)
    constexpr float log2e = 1.4426950408889634f;
    for(int r = 0; r < St.n_rows(); r++) St[r] = cm_exp((St[r] - new_max_t)*log2e);

    vector<float, cols> row_sum_t;
    row_sum_t = cm_add<float>(St[0], St[1]);
    for(int r = 2; r < St.n_rows(); r++) row_sum_t = cm_add<float>(row_sum_t, St[r]);

    vector<float, cols> max_comp;
    max_comp = cm_exp((cur_max - new_max_t)*log2e);
    cur_sum = cm_mul<float>(cur_sum, max_comp);
    cur_sum = cm_add<float>(cur_sum, row_sum_t);
    cur_max = new_max_t;
    return max_comp;
}

//===============================================================================================
template <int i, int N, int M>
constexpr void apply_causal_mask(matrix_ref<float, N, M> St) {
    if constexpr (i < N) {
        St.row(i).select<i, 1>(0) = -3.4e38f;
        apply_causal_mask<i + 1>(St);
    }
}

#ifdef CM_HAS_LSC_UNTYPED_2D
template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused = 0>
void sdpa_kernel_lsc(
    uint slm_K,
    uint slm_V,
    int wg_local_id,
    int local_size,
    int q_start,
    int kv_stop,
    int q_len,
    int kv_len,
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t o_base [[type("svmptr_t")]]) {

    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    constexpr uint kv_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;

    auto q_tokens_left = q_len;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    }

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    int causal_left = q_start;

    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    auto load_slm_KV = [&](int kv_pos) {
        if (kv_pos < kv_stop) {
            uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
            slm_buff_id_write ++;
            if (wg_local_id < local_size/2) {
                vector<half, kv_step * REG_K> temp0;
                b2dK.set_block_y(kv_pos);
                for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                    cm_load<lsc::Normal>(temp0, b2dK.set_block_x(k));
                    cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), temp0);
                }
            } else {
                vector<half, REG_K*REG_N> temp2;
                b2dV.set_block_y(kv_pos);
                #pragma unroll
                for(int k = REG_N*(wg_local_id-(local_size/2)); k < head_size; k += REG_N*(local_size/2)) {
                    cm_load<lsc::VNNI>(temp2, b2dV.set_block_x(k));
                    cm_slm_block_write(slm_V, slm_offset + k * REG_K * sizeof(half), temp2);
                }
            }
        }
    };
    load_slm_KV(0);
    load_slm_KV(kv_step);
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            k_base += kv_step * kv_pitch,
            v_base += kv_step * kv_pitch,
            slm_buff_id_read ++) {

        //  load0, load1, signal1, 
        //  [wait2, signal2, load2, read0]
        //  [wait3, signal3, load3, read1]
        //  [wait4, signal4, load4, read2]  
        //  [wait5, signal5, load5, read3]  
        //
        //  after wait4, all workers have reached signal3, so:
        //     - all workers have finished load2 & read0. 
        //     - we can start to load 4 into SLM slot 0 (i & 3) safely 
        //     - we can start to read 2 ((i-2) & 3) safely

        cm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(0);
        //if (kv_pos > 1024000) // for debugging
        if (kv_pos + kv_step < kv_stop)
            cm_sbarrier(1);

        load_slm_KV(kv_pos + kv_step*2);

        {
            uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;
            //# St = k @ Qt
            matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

            if constexpr (use_causal_mask) {
                // since kv_step == q_step == 16, causal_left is n*kv_step
                if (causal_left == 0) {
                    apply_causal_mask<1>(St);
                } else if (causal_left < 0) {
                    St = -3.4e38f;
                }
                causal_left -= kv_step;
            } else {
                int kv_tokens = kv_stop - kv_pos;
                // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
                for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
            }

            //show(St);
            auto max_comp = online_softmax_update(St, cur_max, cur_sum);

            matrix<half, REG_N, REG_K> P;
            Transpose2DMatrix(St, P);

            if (kv_pos == 0)
                ugemm_PV0(slm_V, P, rO, slm_offset);
            else
                ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
        }
    }
    // cm_sbarrier(0);
    if (q_tokens_left == 0) return;

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);

    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
            }
        }
        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}


template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
void sdpa_kernel_lsc_prefetch(
    int wg_local_id,
    int q_start,
    int kv_stop,
    int q_len,
    int kv_len,
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t o_base [[type("svmptr_t")]]) {

    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    constexpr uint kv_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;

    auto q_tokens_left = q_len;// - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    }

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    int causal_left = q_start;

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            k_base += kv_step * kv_pitch,
            v_base += kv_step * kv_pitch) {
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St; // = ugemm_KQ(slm_K, rQ, slm_offset);
        {
            constexpr int num_K = kv_step/REG_M;
            auto St2 = St.format<float, num_K, REG_M*REG_N>();

            matrix<half, num_K, REG_M * REG_K> Kmat;
            //cm_slm_block_read(slm_K, GENX_NONE, slm_offset, Kmat.format<half>());
            prefetch_K.set_block_y(wg_local_id + kv_pos + kv_step);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));

            b2dK.set_block_y(kv_pos);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            #pragma unroll
            for(int k = 0; k < num_K; k++)
                St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0,
                                rQ[0].format<int32_t>(),
                                Kmat[k].format<int32_t>());

            #pragma unroll
            for(int ri = 1; ri < head_size/REG_K; ri++) {
                //cm_slm_block_read(slm_K, GENX_NONE, slm_offset + ri * Kmat.n_elems() * sizeof(half), Kmat.format<half>());
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri*REG_K));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(ri*REG_K));
                #pragma unroll
                for(int k = 0; k < num_K; k++) {
                    St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St2.row(k),
                        rQ[ri].format<int32_t>(),
                        Kmat[k].format<int32_t>());
                }
            }
        }
        if constexpr (use_causal_mask) {
            // since kv_step == q_step == 16, causal_left is n*kv_step
            if (causal_left == 0) {
                apply_causal_mask<1>(St);
            } else if (causal_left < 0) {
                St = -3.4e38f;
            }
            causal_left -= kv_step;
        } else {
            int kv_tokens = kv_stop - kv_pos;
            // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
            for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
        }

        //show(St);
        auto max_comp = online_softmax_update(St, cur_max, cur_sum);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        b2dV.set_block_y(kv_pos);
        prefetch_V.set_block_y(wg_local_id +kv_pos + kv_step);
        if (kv_pos == 0) {
            // ugemm_PV0(slm_V, P, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
        }
        else {
            //ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));

                //# compensate cur_O
                //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
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
    }
    if (q_tokens_left == 0) return;

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);

    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
            }
        }
        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}
#endif

template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused = 0>
void sdpa_kernel(
    uint slm_K,
    uint slm_V,
    int wg_local_id,
    int local_size,
    int q_start,
    int kv_stop,
    int q_len,
    int kv_len,
    SurfaceIndex query [[type("buffer_t")]],
    SurfaceIndex key [[type("buffer_t")]],
    SurfaceIndex value [[type("buffer_t")]],
    SurfaceIndex output [[type("buffer_t")]],
    uint q_off,
    uint k_off,
    uint v_off,
    uint o_off) {
    constexpr uint padded_head_size = (head_size + 16 - 1) / 16 *16;
    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    constexpr uint kv_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;

    matrix<half, padded_head_size/REG_K, REG_K*REG_N> rQ;
    auto q_tokens_left = q_len;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);
    static_assert((head_size % 8) == 0);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        // load as many as possible given one address
        if constexpr (head_size == 128 || head_size == 64) {
            matrix<uint, q_step, head_size/2> QmatI32;
            cm_load_2d(QmatI32, query, q_off, q_pitch);
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
                Transpose2DMatrix(QmatI32.select<q_step, 1, REG_K/2, 1>(0, k), rQ[ri].format<uint, REG_K/2, q_step>());
                rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
            }
        } else {
            constexpr uint num_blocks = (padded_head_size + REG_K - 1) / REG_K;
            #pragma unroll
            for(int i = 0, ri = 0; i < num_blocks; i += 1, ri++) {
                matrix<uint, q_step, REG_K/2> QmatI32;
                int k = i * REG_K;
                int valid_cols = head_size - k;
                if (valid_cols >= REG_K) {
                    cm_load_2d(QmatI32, query, q_off + k * sizeof(uint) / 2, q_pitch);
                } else {
                    // QmatI32 uses int so that head_size_tail(half) should be divided by 2
                    cm_load_2d_with_tail<q_step, REG_K/2, (head_size % REG_K) / 2>(QmatI32, query, q_off + k * sizeof(half), q_pitch);
                }
                Transpose2DMatrix(QmatI32, rQ[ri].format<uint, REG_K/2, q_step>());
                rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
            }
        }
    }

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix <float, padded_head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    int causal_left = q_start;

    constexpr uint slm_buff_size = kv_step * padded_head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    auto load_slm_KV = [&](int kv_pos) {
        //if (kv_pos < 1024000) return;
        int kv_tokens = kv_stop - kv_pos;
        if (kv_tokens <= 0) return;
        uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
        slm_buff_id_write ++;

        // non-tail branch is faster
        if (wg_local_id < local_size/2) {
            //if (kv_pos > 1024000) {
            matrix<half, 2*REG_M, REG_K> temp;
            // head_size is split into blocks of <half, REG_K>
            constexpr uint num_full_blocks = head_size / REG_K;
            int i = wg_local_id;
            for (; i < num_full_blocks; i += local_size / 2) {
                int k = i * REG_K;
                cm_load_2d(temp, key, k_off + k * sizeof(half), kv_pitch);
                cm_slm_block_write(slm_K, 
                    slm_offset + k * 2 * REG_M * sizeof(half),
                    temp.format<half>());
            }
            // if with tail, load with head_size % REG_K
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % REG_K > 0) {
                if (i == num_full_blocks) {
                    int k = num_full_blocks * REG_K;
                    cm_load_2d_with_tail<2*REG_M, REG_K, head_size % REG_K>(temp, key, k_off + k * sizeof(half), kv_pitch);
                    cm_slm_block_write(slm_K, 
                        slm_offset + k * 2 * REG_M * sizeof(half),
                        temp.format<half>());
                }
                i += local_size / 2;
            }
        } else {
            //if (kv_pos > 1024000) {
            // read 16x16 XMX-B matrix (1x REG_N in Xe2, 2x REG_N in Xe1)
            constexpr int VK_STEP = 16;
            static_assert((VK_STEP % REG_N) == 0);
            matrix<half, REG_K, VK_STEP> temp2;
            matrix<half, REG_K/2, REG_N*2> temp_vnni;
            //b2dV.set_block_y(kv_pos);

            // head_size is split into blocks of <half, VK_STEP>
            constexpr uint num_full_blocks = head_size / VK_STEP;
            int i = wg_local_id - local_size / 2;
            for (; i < num_full_blocks; i += local_size / 2) {
                int k = i * VK_STEP;
                cm_load_2d(temp2, value, v_off + k * sizeof(half), kv_pitch);
                #pragma unroll
                for (int p = 0; p < VK_STEP / REG_N; p++) {
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K / 2, 2, REG_N, 1>(0, p * REG_N);
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K / 2, 2, REG_N, 1>(1, p * REG_N);

                    cm_slm_block_write(slm_V, slm_offset + (k + p * REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                }
            }
            // if with tail, load with head_size_tail
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % REG_K > 0) {
                if (i == num_full_blocks) {
                    int k = num_full_blocks * VK_STEP;
                    cm_load_2d_with_tail<REG_K, VK_STEP, head_size % REG_K>(temp2, value, v_off + k * sizeof(half), kv_pitch);
                    #pragma unroll
                    for (int p = 0; p < VK_STEP / REG_N; p++) {
                        temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K / 2, 2, REG_N, 1>(0, p * REG_N);
                        temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K / 2, 2, REG_N, 1>(1, p * REG_N);

                        cm_slm_block_write(slm_V, slm_offset + (k + p * REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                    }
                    i += local_size / 2;
                }
            }
        }
        k_off += kv_step * kv_pitch;
        v_off += kv_step * kv_pitch;
        // printf(" diff= %lu\n", get_clock() - clk0);
    };

    load_slm_KV(0);
    load_slm_KV(kv_step);

    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            slm_buff_id_read ++) {
        //
        //  load0->0, signal1, 
        //  [load1->1, wait2, signal2, read0]
        //  [load2->2, wait3, signal3, read1]
        //  [load3->3, wait4, signal4, read2]  
        //  [load4->0, wait5, signal5, read3]  
        //
        //  after wait4, all workers have reached signal3, so:
        //     - all workers have finished load2 & read0. 
        //     - we can start to load 4 into SLM slot 0 (i & 3) safely 
        //     - we can start to read 2 ((i-2) & 3) safely
        //
        cm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(0);

        load_slm_KV(kv_pos + 2*kv_step);

        if (kv_pos + kv_step < kv_stop)
            cm_sbarrier(1);

        //if (kv_pos < 1024000) continue;
        uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;

        //=========================================================== 1807 ~ 3247
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

        if constexpr (use_causal_mask) {
            if (causal_left < kv_step) {
                vector<float, q_step> cmask = 0.0f;
                int p = causal_left + 1;
                int v = 0;
                for(; p < 0; p++) {
                    cmask[v] = -3.4e38f;
                    if (v < q_step - 1) v++;
                }
                for(; p < kv_step; p++) {
                    cmask[v] = -3.4e38f;
                    St[p] = cm_add<float>(St[p], cmask);
                    if (v < q_step - 1) v++;
                }
                //if (wg_local_id == 0) show(St);return;
            }
            causal_left -= kv_step;
        }

        // mask off k-tails
        int kv_tokens = kv_stop - kv_pos;
        for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;

        //show(St);
        auto max_comp = online_softmax_update(St, cur_max, cur_sum);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        if (kv_pos == 0)
            ugemm_PV0(slm_V, P, rO, slm_offset);
        else
            ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
    }

    if (q_tokens_left > 0) {
        //# save cur_O/cur_sum.transpose(0, 1)
        matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
        cur_sum = cm_inv(cur_sum);

        #pragma unroll
        for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
            #pragma unroll
            for(int p = 0; p < num_P_tiles; p++) {
                auto cO = rO[ri + p].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < cO.n_rows(); r++) {
                    cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
                }
            }
            // if (i == args_verbose) show(cur_O_f16);
            cm_store_2d(cur_O_f16, output, o_off + k*sizeof(half), o_pitch);
        }
    }
}
