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
#include "cm_attention_common.hpp"

#ifdef CM_HAS_LSC_UNTYPED_2D
#define USE_LSC 1
#else
#define USE_LSC 0
#endif

#if CMPA_KVCACHE_U8
template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_q_fused = 0>
void pa_lsc_u8(
    uint slm_K,
    uint slm_V,
    int wg_local_id,
    int local_size,
    int q_start,
    int kv_stop,
    int q_len,
    int kv_len,
#if USE_LSC
    svmptr_t q_base [[type("svmptr_t")]],
#else
    SurfaceIndex q_gather,
    uint32_t q_gather_offset_bytes,
#endif
    svmptr_t k_cache_base [[type("svmptr_t")]],
    svmptr_t v_cache_base [[type("svmptr_t")]],
#if IS_BLOCK_SPARSE
    svmptr_t sparse_mask_base [[type("svmptr_t")]],
    svmptr_t wg_sparse_mask_base [[type("svmptr_t")]],
    int SPARSE_BLOCK_SIZE,
#endif
    svmptr_t o_base [[type("svmptr_t")]],
    int32_t past_lens,
    int32_t* block_indices [[type("svmptr_t")]]) {

    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_q_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    //[block_num, kv_heads, block_size, head_size]
    constexpr uint kv_pitch = head_size * sizeof(uint8_t);

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
        #if USE_LSC
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
        #else
        constexpr int q_tile_uints  = REG_K / 2;
        constexpr int q_tile_elems  = q_tile_uints * REG_N;
        vector<ushort, q_tile_elems> gather_pred;

        #pragma unroll
        for (int ri = 0; ri < head_size/REG_K; ri++) {
            vector<unsigned, q_tile_elems> gather_offsets;
            uint col_uint_base = ri * q_tile_uints;

            #pragma unroll
            for (int col = 0; col < REG_N; col++) {
                bool active     = (col < q_tokens_left);
                uint token_base = q_gather_offset_bytes + col * q_pitch;

                #pragma unroll
                for (int row = 0; row < q_tile_uints; row++) {
                    int idx      = row * REG_N + col;
                    uint col_byte = (col_uint_base + row) * sizeof(uint);
                    gather_offsets[idx] = token_base + col_byte;
                    gather_pred[idx]    = active ? 0xFFFF : 0;
                }
            }

            rQ[ri] = 0;
            auto gathered = cm_load<uint,
                                    VectorSize::N1,
                                    DataSize::U32,
                                    CacheHint::Cached,
                                    CacheHint::Cached>(q_gather, gather_offsets, gather_pred);
            rQ[ri].format<uint>()  = gathered;
            rQ[ri].format<half>()  = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
        #endif
    }
    #if USE_LSC
    lsc::block_2d_desc<uint8_t, 1, kv_step, REG_K> b2dK(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<uint8_t, 1, REG_K, REG_N> b2dV(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
    #endif
    constexpr int quan_blk_stride = CMFLA_NUM_KV_HEADS * (CMFLA_HEAD_SIZE+4) * CMPA_BLOCK_SZ * sizeof(uint8_t);
    int causal_left = q_start + past_lens;

    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

#if IS_BLOCK_SPARSE
#if USE_LSC
    auto skip_compute = [&](int kv_pos) {
        auto kv_start_block = kv_pos / SPARSE_BLOCK_SIZE;
        bool sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);

        return !sparse_mask;
    };
    auto skip_load = [&](int kv_pos) {
        auto kv_start_block = kv_pos / SPARSE_BLOCK_SIZE;
        bool sparse_mask = *(reinterpret_cast<bool*>(wg_sparse_mask_base) + kv_start_block);
        return !sparse_mask;
    };
#else
    auto skip_compute = [&](int kv_pos) {
        uint kv_start_block = 0;
        bool sparse_mask = true;
        if (SPARSE_BLOCK_SIZE == 64) {
            kv_start_block = (uint)kv_pos >> 6;
            sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
        } else if (SPARSE_BLOCK_SIZE == 128) {
            kv_start_block = (uint)kv_pos >> 7;
            sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
        } else if (SPARSE_BLOCK_SIZE == 256) {
            kv_start_block = (uint)kv_pos >> 8;
            sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
        } else {
            sparse_mask = true;
        }
        return !sparse_mask;
    };
    auto skip_load = [&](int kv_pos) {
        uint kv_start_block = 0;
        bool sparse_mask = true;
        if (SPARSE_BLOCK_SIZE == 64) {
            kv_start_block = (uint)kv_pos >> 6;
            sparse_mask = *(reinterpret_cast<bool*>(wg_sparse_mask_base) + kv_start_block);
        } else if (SPARSE_BLOCK_SIZE == 128) {
            kv_start_block = (uint)kv_pos >> 7;
            sparse_mask = *(reinterpret_cast<bool*>(wg_sparse_mask_base) + kv_start_block);
        } else if (SPARSE_BLOCK_SIZE == 256) {
            kv_start_block = (uint)kv_pos >> 8;
            sparse_mask = *(reinterpret_cast<bool*>(wg_sparse_mask_base) + kv_start_block);
        } else {
            sparse_mask = true;
        }
        return !sparse_mask;
    };
#endif
#endif

    auto load_slm_KV = [&](int kv_pos) {
        if (kv_pos < kv_stop) {
#if IS_BLOCK_SPARSE
            if (SPARSE_BLOCK_SIZE > 1 && skip_load(kv_pos)) {
                slm_buff_id_write++;
                return;
            }
#endif
            auto cur_block_id = block_indices[kv_pos / CMPA_BLOCK_SZ];
            uint32_t dscale_offset = cur_block_id*quan_blk_stride + \
                        CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) + kv_pos%CMPA_BLOCK_SZ*sizeof(half);

            uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
            vector<half, kv_step> dscale;
            vector<half, kv_step> zp;
            int kv_left =  (kv_stop-kv_pos) > kv_step ? kv_step: (kv_stop-kv_pos);

            slm_buff_id_write ++;
            if (wg_local_id < local_size/2) {
                cm_svm_block_read(reinterpret_cast<svmptr_t>( k_cache_base + dscale_offset), dscale);
                cm_svm_block_read(reinterpret_cast<svmptr_t>( k_cache_base + dscale_offset + CMPA_BLOCK_SZ*sizeof(half)), zp);

                matrix<half, kv_step, REG_K> kmat;
                auto quanKmat = kmat.format<half, 2, kv_step * REG_K/2>()[1].format<uint8_t, kv_step, REG_K>();
                #if USE_LSC
                b2dK.set_base_ptr(reinterpret_cast<uint8_t*>(k_cache_base+cur_block_id*quan_blk_stride));
                b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);
                #endif
                
                // This condition only works for head_size <= 128
                for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                    #if USE_LSC
                    cm_load<lsc::Normal>(quanKmat.format<uint8_t>(), b2dK.set_block_x(k));
                    #else
                    auto k_base = reinterpret_cast<svmptr_t>((int8_t*)k_cache_base + cur_block_id * quan_blk_stride + (kv_pos % CMPA_BLOCK_SZ) * kv_pitch + k);
                    #pragma unroll
                    for(int r = 0; r < kv_step; r++) {
                        cm_svm_block_read(k_base + r * kv_pitch, quanKmat.row(r));
                    }
                    #endif
                    /*@bug: cm compiler in the tail process.
                          :  loop combined with type convert.
                        for(int r = 0; r < kv_left; r++) {
                            kmat[r] =  quanKmat[r]-zp[r];
                            kmat[r] = cm_mul<half>(kmat[r], dscale[r]);
                        }
                     wa: unroll all kv_step rows. set 0 to padding rows.
                    */
                    #pragma unroll
                    for(int r = 0; r < kv_step; r++)  {
                        kmat[r] =  quanKmat[r]-zp[r];
                        kmat[r] = cm_mul<half>(kmat[r], dscale[r]);
                    }
                    //clear unused data to 0.
                    for(int r = kv_step-1; r >= kv_left; r--)
                        kmat[r] = 0;
                    cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), kmat.format<half>());
                }
            } else {
                cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + dscale_offset), dscale);
                cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + dscale_offset+CMPA_BLOCK_SZ*sizeof(half)), zp);

                matrix<half, REG_K/2, REG_N*2> VmatVNNI;
                matrix<half, REG_K, REG_N> Vmat;
                auto quanVmat = Vmat.format<half, 2, REG_K*REG_N/2>().row(1).format<uint8_t, REG_K, REG_N>();
                #if USE_LSC
                b2dV.set_base_ptr(reinterpret_cast<uint8_t*>(v_cache_base+cur_block_id*quan_blk_stride));
                b2dV.set_block_y(kv_pos%CMPA_BLOCK_SZ);
                #endif
                #pragma unroll
                for(int k = REG_N*(wg_local_id-(local_size/2)); k < head_size; k += REG_N*(local_size/2)) {
                    #if USE_LSC
                    cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(k));
                    #else
                    auto v_base = reinterpret_cast<svmptr_t>((int8_t*)v_cache_base + cur_block_id * quan_blk_stride + (kv_pos % CMPA_BLOCK_SZ) * kv_pitch + k);
                    #pragma unroll
                    for(int r = 0; r < REG_K; r++) {
                        cm_svm_block_read(v_base + r * kv_pitch, quanVmat.row(r));
                    }
                    #endif
                    /*@bug: cm compiler in the tail process.
                          :  loop combined with type convert.
                        for(int r = 0; r < kv_left; r++) {
                            Vmat[r] =  quanVmat[r]-zp[r];
                            Vmat[r] = cm_mul<half>(Vmat[r], dscale[r]);
                        }
                    */
                    #pragma unroll
                    for(int r = 0; r < kv_step;r++)  {
                        Vmat[r] =  quanVmat[r]-zp[r];
                        Vmat[r] = cm_mul<half>(Vmat[r], dscale[r]);
                    }

                    for(int r = kv_step-1; r>=kv_left;r--)  {
                        Vmat[r] = 0;
                    }
                    prepackAsVNNIWidth2(Vmat, VmatVNNI);
                    cm_slm_block_write(slm_V, slm_offset + k * REG_K * sizeof(half), VmatVNNI.format<half>());
                }
            }
        }
    };

    load_slm_KV(0);
    load_slm_KV(kv_step);
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,slm_buff_id_read++) {

        //  load0, load1, signal1,
        //  [wait1, signal2, load2, read0, compute0]
        //  [wait2, signal3, load3, read1, compute1]
        //  [wait3, signal4, load4, read2, compute2]
        //  [wait4, signal5, load5, read3, compute3]
        //
        //  after wait3, all workers have reached signal3, so:
        //     - all workers have finished load2 & read0.
        //     - we can start to load 4 into SLM slot 0 (i & 3) safely
        //     - we can start to read 2 ((i-2) & 3) safely


        cm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(0);
        //if (kv_pos > 1024000)
        if (kv_pos + kv_step < kv_stop)
            cm_sbarrier(1);
        load_slm_KV(kv_pos + kv_step*2);


#if IS_BLOCK_SPARSE
        if (SPARSE_BLOCK_SIZE > 1 && skip_compute(kv_pos)) {
            if constexpr (use_causal_mask)
                causal_left -= kv_step;
            continue;
        }
#endif
        {

            uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;

            //# St = k @ Qt
            matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);
            if constexpr (use_causal_mask) {
                #if kv_step == q_step
                // since kv_step == q_step == 16, causal_left is n * kv_step
                if (causal_left == 0) {
                    apply_causal_mask<1>(St);
                } else if (causal_left < 0) {
                    St = -3.4e38f;
                }
                #else
                if (causal_left == 0) {
                    // q_step is half of kv_step
                    // calsual mask first half of the kv
                    apply_causal_mask<1>(St.select<q_step, 1, q_step, 1>(0, 0));
                    St.select<q_step, 1, q_step, 1>(q_step, 0) = -3.4e38f;
                } else if (causal_left < 0) {
                    St = -3.4e38f;
                } else if (causal_left < kv_step) {
                    // q_step is half of kv_step
                    // calsual mask second half of the kv
                    // if w/o St += 0.f;, I will meet IGC: Internal Compiler Error: Access violation on ARL-H
                    St += 0.f;
                    apply_causal_mask<1>(St.select<q_step, 1, q_step, 1>(q_step, 0));
                }
                #endif
                causal_left -= kv_step;
            } else {
                int kv_tokens = kv_stop - kv_pos;
                // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
                for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
            }
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

    #if USE_LSC
    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);
    #endif
    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        #if USE_LSC
        b2dO.set_block_x(k);
        #endif
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
            }
            #if USE_LSC
            cm_store(b2dO.set_block_y(p * REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(p));
            #else
            int o_stride_elems = o_pitch / sizeof(half);
            half* output_ptr = (half*)o_base + p * REG_M * o_stride_elems + k;
            auto cur_O_ref = cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(p).format<half, REG_M, REG_N>();
            #pragma unroll
            for (int r = 0; r < REG_M; r++) {
                cm_svm_block_write<half, REG_N>((svmptr_t)(output_ptr + r * o_stride_elems), cur_O_ref.row(r).format<half>());
            }
            #endif
        }
    }
}

#else

template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
void pa_kernel_lsc_prefetch_f16(
    int wg_local_id,
    int q_start,
    int kv_stop, //
    int q_len, //q_step
    int kv_len, //not used for now
#if USE_LSC
    svmptr_t q_base [[type("svmptr_t")]],
#else
    SurfaceIndex q_gather,
    uint32_t q_gather_offset_bytes,
#endif
    svmptr_t k_cache_base [[type("svmptr_t")]],
#if USE_LSC
    svmptr_t v_cache_base [[type("svmptr_t")]],
    #else
    SurfaceIndex v_cache_stateful,
    uint32_t v_cache_stateful_offset_bytes,
#endif
#if IS_BLOCK_SPARSE
    svmptr_t sparse_mask_base [[type("svmptr_t")]],
    svmptr_t wg_sparse_mask_base [[type("svmptr_t")]],
    int SPARSE_BLOCK_SIZE,
#endif
    svmptr_t o_base [[type("svmptr_t")]],
    int32_t past_lens,
    int32_t* block_indices [[type("svmptr_t")]]) {
    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    // constexpr uint k_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));
    // constexpr uint v_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));
    //[block_num, kv_heads, block_size, head_size]
    constexpr uint k_pitch = head_size * sizeof(half);
    constexpr uint v_pitch = k_pitch;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
#if USE_LSC
    constexpr int VALUE_TILE_NUM = 1;
#else
    constexpr int VALUE_TILE_NUM = 2;
#endif
    matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
    matrix <float, head_size/REG_M, REG_M*REG_N> rO;

    auto q_tokens_left = q_len;// - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;
    if (q_tokens_left == 0) return;

    if (q_tokens_left > 0) {
        #if USE_LSC
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
        #else
        constexpr int q_tile_uints = REG_K / 2;
        constexpr int q_tile_elems = q_tile_uints * REG_N;
        vector<ushort,  q_tile_elems> gather_pred;

        #pragma unroll
        for (int ri = 0; ri < head_size/REG_K; ri++) {
            vector<unsigned, q_tile_elems> gather_offsets;
            uint col_uint_base = ri * q_tile_uints;
            #pragma unroll
            for (int col = 0; col < REG_N; col++) {
                bool active = (col < q_tokens_left);
                uint token_base = q_gather_offset_bytes + col * q_pitch;
                #pragma unroll
                for (int row = 0; row < q_tile_uints; row++) {
                    int idx = row * REG_N + col;
                    uint col_byte = (col_uint_base + row) * sizeof(uint);
                    gather_offsets[idx] = token_base + col_byte;
                    gather_pred[idx]    = active ? 0xFFFF : 0;
                }
            }
            rQ[ri] = 0;
            auto gathered =cm_load<uint,
                        VectorSize::N1,
                        DataSize::U32,
                        CacheHint::Cached,
                        CacheHint::Cached>(q_gather, gather_offsets, gather_pred);
            rQ[ri].format<uint>()  = gathered;
            rQ[ri].format<half>()  = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
        #endif
    }

    #if USE_LSC
    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);
    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);
    #endif
    constexpr int blk_stride = CMFLA_NUM_KV_HEADS * CMFLA_HEAD_SIZE*CMPA_BLOCK_SZ;
    int causal_left = q_start+past_lens;

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step) {
        auto cur_block_id = block_indices[kv_pos / CMPA_BLOCK_SZ];
        //For the last step, duplicate prefetch here.
        uint32_t prefetch_kv_pos = (kv_pos+kv_step) >= kv_stop ?  kv_pos : (kv_pos+kv_step);
        auto prefetch_block_id = block_indices[prefetch_kv_pos / CMPA_BLOCK_SZ];
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St;
        {
            constexpr int num_K = kv_step / REG_M;
            auto St2 = St.format<float, num_K, REG_M*REG_N>();

            matrix<half, num_K, REG_M * REG_K> Kmat;

            #if USE_LSC
            prefetch_K.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+prefetch_block_id * blk_stride));
            prefetch_K.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
            #else
            half* prefetch_k_pos = (half*)k_cache_base + prefetch_block_id * blk_stride + ((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ) * head_size;
            cm_ptr_prefetch<REG_K/2, DataSize::U32, CacheHint::Cached, CacheHint::Cached>((const unsigned int *const)prefetch_k_pos, 0);
            #endif

#if IS_BLOCK_SPARSE
            if (SPARSE_BLOCK_SIZE > 1)
            {
            #if USE_LSC
                auto kv_start_block = kv_pos / SPARSE_BLOCK_SIZE;
                bool sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
            #else
                uint kv_start_block = 0;
                bool sparse_mask = true;
                if (SPARSE_BLOCK_SIZE == 64) {
                    kv_start_block = (uint)kv_pos >> 6;
                    sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
                } else if (SPARSE_BLOCK_SIZE == 128) {
                    kv_start_block = (uint)kv_pos >> 7;
                    sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
                } else if (SPARSE_BLOCK_SIZE == 256) {
                    kv_start_block = (uint)kv_pos >> 8;
                    sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
                } else {
                    sparse_mask = true;
                }
            #endif
                if (!sparse_mask) {
                    if constexpr (use_causal_mask) {
                        causal_left -= kv_step;
                    }
                    continue;
                }
            }
#endif
            #if USE_LSC
            b2dK.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+cur_block_id*blk_stride));
            b2dK.set_block_y(kv_pos % CMPA_BLOCK_SZ);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            #else
            half* base_k_cache_ptr = (half*)k_cache_base + cur_block_id*blk_stride + (kv_pos % CMPA_BLOCK_SZ)*head_size;
            auto kmatref = Kmat.format<half, kv_step, REG_K>();
            #pragma unroll
            for(int kr = 0; kr < kv_step; kr++){
                cm_svm_block_read<half, REG_K>((svmptr_t)((half*)base_k_cache_ptr + kr * head_size), kmatref.row(kr));
            }
            #endif
            // somtimes KV cache would be filled with random Nan, so need to clean up the unused key data.
            if ((kv_pos + kv_step) > kv_stop) {
                auto valid_rows = kv_stop - kv_pos;
                for (int r = valid_rows; r < kv_step; r++)
                    Kmat.format<half, num_K * REG_M, REG_K>().row(r) = 0.f;
            }
            #pragma unroll
            for(int k = 0; k < num_K; k++)
                St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0,
                                rQ[0].format<int32_t>(),
                                Kmat[k].format<int32_t>());

            #pragma unroll
            for(int ri = 1; ri < head_size/REG_K; ri++) {
                #if USE_LSC
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri*REG_K));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(ri*REG_K));
                #else
                cm_ptr_prefetch<REG_K/2, DataSize::U32, CacheHint::Cached, CacheHint::Cached>((const unsigned int *const)prefetch_k_pos, ri*REG_K/2);
                #pragma unroll
                for(int kr = 0; kr < kv_step; kr++){
                    cm_svm_block_read<half, REG_K>((svmptr_t)((half*)base_k_cache_ptr + kr * head_size + ri * REG_K), kmatref.row(kr));
                }
                #endif
                // somtimes KV cache would be filled with random Nan, so need to clean up the unused key data.
                if ((kv_pos + kv_step) > kv_stop) {
                    auto valid_rows = kv_stop - kv_pos;
                    for (int r = valid_rows; r < kv_step; r++)
                        Kmat.format<half, num_K * REG_M, REG_K>().row(r) = 0.f;
                }
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
            #if kv_step == q_step
            // since kv_step == q_step == 16, causal_left is n * kv_step
            if (causal_left == 0) {
                apply_causal_mask<1>(St);
            } else if (causal_left < 0) {
                St = -3.4e38f;
            }
            #else
            if (causal_left == 0) {
                // q_step is half of kv_step
                // calsual mask first half of the kv
                apply_causal_mask<1>(St.select<q_step, 1, q_step, 1>(0, 0));
                St.select<q_step, 1, q_step, 1>(q_step, 0) = -3.4e38f;
            } else if (causal_left < 0) {
                St = -3.4e38f;
            } else if (causal_left < kv_step) {
                // q_step is half of kv_step
                // calsual mask second half of the kv
                // if w/o St += 0.f;, I will meet IGC: Internal Compiler Error: Access violation on ARL-H
                St += 0.f;
                apply_causal_mask<1>(St.select<q_step, 1, q_step, 1>(q_step, 0));
            }
            #endif
            causal_left -= kv_step;
        } else {
            int kv_tokens = kv_stop - kv_pos;
            // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
            for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
        }

        // show(St);
        auto max_comp = online_softmax_update(St, cur_max, cur_sum);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        #if USE_LSC
        prefetch_V.set_base_ptr((reinterpret_cast<half*>(v_cache_base)+prefetch_block_id*blk_stride));
        prefetch_V.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);

        b2dV.set_base_ptr((reinterpret_cast<half*>(v_cache_base)+cur_block_id*blk_stride));
        b2dV.set_block_y(kv_pos%CMPA_BLOCK_SZ);
        #endif
        auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
        matrix<half, REG_K/2, REG_N*2*VALUE_TILE_NUM> Vmat;
        #pragma unroll
        for(int k = 0, ri=0; k < head_size; k += REG_N * VALUE_TILE_NUM, ri += num_P_tiles * VALUE_TILE_NUM) {
            #if USE_LSC
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
            cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
            #else
            matrix<half, REG_K, REG_N*VALUE_TILE_NUM> Vmat_tmp;
            constexpr uint elem_size = sizeof(half);
            constexpr int value_row_u32 = (REG_N * VALUE_TILE_NUM * sizeof(half)) / sizeof(uint);
            #pragma unroll
            for(int Vr = 0; Vr < REG_K; Vr++){
                uint elem_offset = cur_block_id * blk_stride
                                + (kv_pos % CMPA_BLOCK_SZ) * head_size
                                + Vr * head_size
                                + k;
                uint cur_row_offset = v_cache_stateful_offset_bytes + elem_offset * elem_size;
                auto row_vec_u32 = cm_load<uint, value_row_u32>(v_cache_stateful, cur_row_offset);
                Vmat_tmp.row(Vr).format<uint>() = row_vec_u32;
            }
            if ((kv_pos + kv_step) > kv_stop) {
                uint valid_rows = kv_stop - kv_pos;
                for (uint r = valid_rows; r < kv_step; r++)
                    Vmat_tmp.row(r) = 0.f;
            }
            #pragma unroll
            for (int r = 0; r < REG_K/2; r++) {
                Vmat.row(r).select<REG_N*VALUE_TILE_NUM, 2>(0) = Vmat_tmp.row(r*2);
                Vmat.row(r).select<REG_N*VALUE_TILE_NUM, 2>(1) = Vmat_tmp.row(r*2+1);
            }
            #endif
            #if USE_LSC
            // somtimes KV cache would be filled with random Nan, so need to clean up the unused value data.
            if ((kv_pos + kv_step) > kv_stop) {
                uint valid_rows = kv_stop - kv_pos;
                uint valid_rows_vnni = (valid_rows+1)/2;
                for (int r = valid_rows_vnni; r < REG_K/2; r++)
                    Vmat.row(r) = 0.f;
                if (valid_rows % 2 == 1)
                    Vmat.row(valid_rows_vnni-1).select<REG_N, 2>(1) = 0.f;
            }
            #endif

            if (kv_pos == 0) {
                #pragma unroll
                for (int tile = 0; tile < VALUE_TILE_NUM; tile++) {
                    int rO_base = ri + tile * num_P_tiles;
                    auto Vtile = Vmat.format<half, REG_K/2, REG_N*2*VALUE_TILE_NUM>().select<REG_K/2, 1, REG_N*2, 1>(0, REG_N*2*tile);
                    #pragma unroll
                    for (int p = 0; p < num_P_tiles; p++) {
                        rO[rO_base + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                            0,
                            Vtile.format<int32_t>(),
                            P2.row(p).format<int32_t>());
                    }
                }
            } else {
                #pragma unroll
                for (int tile = 0; tile < VALUE_TILE_NUM; tile++) {
                    int rO_base = ri + tile * num_P_tiles;
                    #pragma unroll
                    for(int p = 0; p < num_P_tiles; p++) {
                        auto cO = rO[rO_base + p].format<float, REG_M, REG_N>();
                        #pragma unroll
                        for(int r = 0; r < REG_M; r++)
                            cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                    }
                }

                #pragma unroll
                for (int tile = 0; tile < VALUE_TILE_NUM; tile++) {
                    int rO_base = ri + tile * num_P_tiles;
                    auto Vtile =
                        Vmat.format<half, REG_K/2, REG_N*2*VALUE_TILE_NUM>()
                            .select<REG_K/2, 1, REG_N*2, 1>(0, REG_N*2*tile);
                    #pragma unroll
                    for (int p = 0; p < num_P_tiles; p++) {
                        rO[rO_base + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                            rO[rO_base + p].format<float>(),
                            Vtile.format<int32_t>(),
                            P2.row(p).format<int32_t>());
                    }
                }
            }
        }
    }

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles * REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    #if USE_LSC
    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);
    #endif
    #pragma unroll
    for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
        #if USE_LSC
        b2dO.set_block_x(k);
        #endif
        #pragma unroll
        for(int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();
            #pragma unroll
            for(int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
            }
            #if USE_LSC
            cm_store(b2dO.set_block_y(p * REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(p));
            #else
            int o_stride_elems = o_pitch / sizeof(half);
            half* output_ptr = (half*)o_base + p * REG_M * o_stride_elems + k;
            auto cur_O_ref = cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(p).format<half, REG_M, REG_N>();
            #pragma unroll
            for (int r = 0; r < REG_M; r++) {
                cm_svm_block_write<half, REG_N>((svmptr_t)(output_ptr + r * o_stride_elems), cur_O_ref.row(r).format<half>());
            }
            #endif
        }
    }
}

#endif
