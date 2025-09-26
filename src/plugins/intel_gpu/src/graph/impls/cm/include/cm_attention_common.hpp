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
//# static_assert(__cplusplus >= 202002L);
//# static_assert(__cplusplus >= 202302L);

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

#define DEBUG_SHOW 1
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
    // if (cm_local_id(2) == 3 && cm_group_id(2) == 0) {
    //     show(Kmat.format<half, 16, 16>());
    // }
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
        // if (cm_local_id(2) == 0 && cm_group_id(2) == 0) {
        //     show(Vmat.format<half, 16, 16>());
        // }
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                            0,
                            Vmat.format<int32_t>(),
                            P2.row(p).format<int32_t>());
            //show(rO[ri + p].format<float, REG_M, REG_N>());
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
        // if (cm_local_id(2) == 0 && cm_group_id(2) == 0) {
        //     show(Vmat.format<half, 16, 16>());
        // }
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < REG_M; r++)
                cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
        }

        //show(rO[ri].format<float, REG_M, REG_N>());

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

#ifdef CM_HAS_LSC_UNTYPED_2D
    #define cm_load_normal cm_load<lsc::Normal>
    #define cm_load_transpose cm_load<lsc::Transpose>
    #define cm_load_vnni cm_load<lsc::VNNI>
    #define cm_store_normal cm_store
#else
    // simulation of LSC API using SVM API
    template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
    inline void cm_load_normal(vector_ref<T, NBlocks*BlockH*BlockW> Res, const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, int16_t Pred = 1) {
        static_assert(NBlocks == 1);
        auto pitch = Desc.get_pitch() + 1;
        auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
        #pragma unroll
        for(int i = 0; i < BlockH; i++) {
            cm_svm_block_read(base + i * pitch, Res.select<BlockW, 1>(i*BlockW));
        }
    }

    template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
    inline void cm_load_transpose(vector_ref<T, NBlocks*BlockW*BlockH> Res, const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, int16_t Pred = 1) {
        static_assert(NBlocks == 1);
        auto pitch = Desc.get_pitch() + 1;
        auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
        matrix<T, BlockH, BlockW> temp;
        #pragma unroll
        for(int i = 0; i < BlockH; i++) {
            cm_svm_block_read(base + i * pitch, temp[i]);
        }
        Transpose2DMatrix(temp, Res.format<T, BlockW, BlockH>());
    }

    // in VNNI case, NBlocks is increasing along X dimension (increase cache-line usage)
    template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
    inline void cm_load_vnni(vector_ref<T, NBlocks*BlockW*BlockH> Res, const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, int16_t Pred = 1) {
        static_assert(NBlocks == 1 || NBlocks == 2);
        // each block must be a full XMX B matrix
        static_assert(BlockH == REG_K);
        static_assert(BlockW == REG_N);
        auto pitch = Desc.get_pitch() + 1;
        auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
        matrix<T, BlockH, NBlocks * BlockW> temp;
        #pragma unroll
        for(int i = 0; i < BlockH; i++) {
            cm_svm_block_read(base + i * pitch, temp[i]);
        }

        auto out_vnni = Res.format<T, NBlocks * (BlockH/2), 2*BlockW>();
        #pragma unroll
        for(int i = 0; i < NBlocks; i ++) {
            out_vnni.select<BlockH/2, 1, BlockW, 2>(i*(BlockH/2), 0) = temp.select<BlockH/2, 2, BlockW, 1>(0, i*BlockW);
            out_vnni.select<BlockH/2, 1, BlockW, 2>(i*(BlockH/2), 1) = temp.select<BlockH/2, 2, BlockW, 1>(1, i*BlockW);
        }
    }

    template <typename T = int, unsigned NBlocks = 1, unsigned BlockH = 1, unsigned BlockW = 1>
    inline void cm_store_normal(const lsc::block_2d_desc<T, NBlocks, BlockH, BlockW> &Desc, vector_ref<T, NBlocks*BlockW*BlockH> Res) {
        static_assert(NBlocks == 1);
        auto pitch = Desc.get_pitch() + 1;
        auto base = reinterpret_cast<svmptr_t>(Desc.get_base() + Desc.get_block_y()*pitch + Desc.get_block_x() * sizeof(T));
        #pragma unroll
        for(int i = 0; i < BlockH; i++) {
            cm_svm_block_write(base + i * pitch, Res.select<BlockW, 1>(i*BlockW));
        }
    }
#endif

//===============================================================================================
template <int i, int N, int M>
constexpr void apply_causal_mask(matrix_ref<float, N, M> St) {
    if constexpr (i < N) {
        St.row(i).select<i, 1>(0) = -3.4e38f;
        apply_causal_mask<i + 1>(St);
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
