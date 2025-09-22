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
#include "cm_attention_common.hpp"

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
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_cache_base [[type("svmptr_t")]],
    svmptr_t v_cache_base [[type("svmptr_t")]],
#if SPARSE_BLOCK_SIZE > 1
    svmptr_t sparse_mask_base [[type("svmptr_t")]],
    svmptr_t wg_sparse_mask_base [[type("svmptr_t")]],
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
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    }

    lsc::block_2d_desc<uint8_t, 1, kv_step, REG_K> b2dK(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<uint8_t, 1, REG_K, REG_N> b2dV(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
    constexpr int quan_blk_stride = CMFLA_NUM_KV_HEADS * (CMFLA_HEAD_SIZE+4) * CMPA_BLOCK_SZ * sizeof(uint8_t);
    int causal_left = q_start+past_lens;

    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

#if SPARSE_BLOCK_SIZE > 1
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
#endif

    auto load_slm_KV = [&](int kv_pos) {
        if (kv_pos < kv_stop) {
#if SPARSE_BLOCK_SIZE > 1
            if (skip_load(kv_pos)) {
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
                b2dK.set_base_ptr(reinterpret_cast<uint8_t*>(k_cache_base+cur_block_id*quan_blk_stride));
                b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);

                for(int k = REG_K*wg_local_id; k < head_size; k += REG_K*(local_size/2)) {
                    cm_load<lsc::Normal>(quanKmat.format<uint8_t>(), b2dK.set_block_x(k));
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
                cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base+dscale_offset), dscale);
                cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base+dscale_offset+CMPA_BLOCK_SZ*sizeof(half)), zp);

                matrix<half, REG_K/2, REG_N*2> VmatVNNI;
                matrix<half, REG_K, REG_N> Vmat;
                auto quanVmat = Vmat.format<half, 2, REG_K*REG_N/2>().row(1).format<uint8_t, REG_K, REG_N>();
                b2dV.set_base_ptr(reinterpret_cast<uint8_t*>(v_cache_base+cur_block_id*quan_blk_stride));
                b2dV.set_block_y(kv_pos%CMPA_BLOCK_SZ);

                #pragma unroll
                for(int k = REG_N*(wg_local_id-(local_size/2)); k < head_size; k += REG_N*(local_size/2)) {
                    cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(k));
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


#if SPARSE_BLOCK_SIZE > 1
            if (skip_compute(kv_pos)) {
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

#else

template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
void pa_kernel_lsc_prefetch_f16(
    int wg_local_id,
    int q_start,
    int kv_stop, //
    int q_len, //q_step
    int kv_len, //not used for now
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_cache_base [[type("svmptr_t")]],
    svmptr_t v_cache_base [[type("svmptr_t")]],
#if SPARSE_BLOCK_SIZE > 1
    svmptr_t sparse_mask_base [[type("svmptr_t")]],
    svmptr_t wg_sparse_mask_base [[type("svmptr_t")]],
#endif
    svmptr_t o_base [[type("svmptr_t")]],
    int32_t past_lens,
    int32_t* block_indices [[type("svmptr_t")]]) {
    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    // constexpr uint k_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));
    // constexpr uint v_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));
    //[block_num, kv_heads, block_size, head_size]
    constexpr uint k_pitch =  head_size * sizeof(half);
    constexpr uint v_pitch = k_pitch;

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

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);
    constexpr int blk_stride = CMFLA_NUM_KV_HEADS*CMFLA_HEAD_SIZE*CMPA_BLOCK_SZ;
    int causal_left = q_start+past_lens;

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step) {
        auto cur_block_id = block_indices[kv_pos / CMPA_BLOCK_SZ];
        //For the last step, duplicate prefetch here.
        uint32_t prefetch_kv_pos = (kv_pos+kv_step) >= kv_stop ?  kv_pos : (kv_pos+kv_step);
        auto prefetch_block_id = block_indices[prefetch_kv_pos / CMPA_BLOCK_SZ];
        //# St = k @ Qt
        matrix<float, kv_step, q_step> St;
        {
            constexpr int num_K = kv_step/REG_M;
            auto St2 = St.format<float, num_K, REG_M*REG_N>();

            matrix<half, num_K, REG_M * REG_K> Kmat;

            prefetch_K.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+prefetch_block_id*blk_stride));
            prefetch_K.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));

#if SPARSE_BLOCK_SIZE > 1
            {
                auto kv_start_block = kv_pos/ SPARSE_BLOCK_SIZE;
                bool sparse_mask = *(reinterpret_cast<bool*>(sparse_mask_base) + kv_start_block);
                if (!sparse_mask) {
                    if constexpr (use_causal_mask) {
                        causal_left -= kv_step;
                    }
                    continue;
                }
            }
#endif
            b2dK.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+cur_block_id*blk_stride));
            b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            #pragma unroll
            for(int k = 0; k < num_K; k++)
                St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0,
                                rQ[0].format<int32_t>(),
                                Kmat[k].format<int32_t>());

            #pragma unroll
            for(int ri = 1; ri < head_size/REG_K; ri++) {
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

        prefetch_V.set_base_ptr((reinterpret_cast<half*>(v_cache_base)+prefetch_block_id*blk_stride));
        prefetch_V.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);

        b2dV.set_base_ptr((reinterpret_cast<half*>(v_cache_base)+cur_block_id*blk_stride));
        b2dV.set_block_y(kv_pos%CMPA_BLOCK_SZ);
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
