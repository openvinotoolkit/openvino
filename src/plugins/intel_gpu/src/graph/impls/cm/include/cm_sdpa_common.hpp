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

// Number of kv sub-tiles (kv_step rows each) processed per online-softmax update in
// sdpa_kernel_lsc_prefetch. Larger amortizes the rO rescale + softmax bookkeeping over
// more DPAS work at the cost of GRF pressure. Overridable via -DCMFLA_KV_BLK=N.
#ifndef CMFLA_KV_BLK
#define CMFLA_KV_BLK 1
#endif

// online_softmax_update_tree and transpose_St_to_P_half are defined in cm_attention_common.hpp

#ifdef CM_HAS_LSC_UNTYPED_2D
//@prefetch_u8 would have duplicated decompress perf issue. comments out for now.
// template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
// void sdpa_kernel_lsc_prefetch_u8(
//     int wg_local_id,
//     int q_start,
//     int kv_stop, //
//     int q_len, //q_step
//     int kv_len, //not used for now
//     svmptr_t q_base [[type("svmptr_t")]],
//     svmptr_t k_cache_base [[type("svmptr_t")]],
//     svmptr_t v_cache_base [[type("svmptr_t")]],
//     svmptr_t o_base [[type("svmptr_t")]],
//     int32_t past_lens,
//     int32_t* block_indices [[type("svmptr_t")]]) {
//     constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
//     constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
//     //[block_num, kv_heads, block_size, head_size]
//     constexpr uint kv_pitch =   head_size * sizeof(uint8_t);

//     vector<float, q_step> cur_max;
//     vector<float, q_step> cur_sum;

//     cur_max = -3e38f;
//     cur_sum = 0;
//     constexpr int num_P_tiles = REG_N / REG_M;
//     matrix<half, head_size/REG_K, REG_K*REG_N> rQ;
//     matrix <float, head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;

//     auto q_tokens_left = q_len;// - q_start;
//     static_assert(q_step == REG_N);
//     static_assert(kv_step == REG_K);

//     if (q_tokens_left < 0) q_tokens_left = 0;
//     if (q_tokens_left > q_step) q_tokens_left = q_step;

//     if (q_tokens_left > 0) {
//         lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
//         #pragma unroll
//         for(int k = 0, ri = 0; k < head_size/2; k += REG_K/2, ri++) {
//             cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
//             rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
//         }
//     }

//     lsc::block_2d_desc<uint8_t, 1, kv_step, REG_K> b2dKV(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);

//     static_assert(wg_local_size == 16);
//     lsc::block_2d_desc<uint8_t, 1, kv_step/wg_local_size, REG_K> b2dKV_prefetch(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
//     // constexpr int blk_stride = CMFLA_NUM_KV_HEADS * CMFLA_HEAD_SIZE * CMPA_BLOCK_SZ;
//     constexpr int quan_blk_stride = CMFLA_NUM_KV_HEADS * (CMFLA_HEAD_SIZE+4) * CMPA_BLOCK_SZ * sizeof(uint8_t);



//     int causal_left = q_start+past_lens;
//     for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step) {
//         auto cur_block_id = block_indices[kv_pos / CMPA_BLOCK_SZ];
//         //For the last step, duplicate prefetch here.
//         uint32_t prefetch_kv_pos = (kv_pos+kv_step) >= kv_stop ?  kv_pos : (kv_pos+kv_step);
//         auto prefetch_block_id = block_indices[prefetch_kv_pos / CMPA_BLOCK_SZ];
//         uint32_t dscale_offset = cur_block_id*quan_blk_stride + CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) + kv_pos%CMPA_BLOCK_SZ*sizeof(half);

//         vector<half, kv_step> dscale;
//         vector<half, kv_step> zp;
//         cm_svm_block_read(reinterpret_cast<svmptr_t>( k_cache_base + dscale_offset), dscale);
//         cm_svm_block_read(reinterpret_cast<svmptr_t>( k_cache_base + dscale_offset + CMPA_BLOCK_SZ*sizeof(half)), zp);

//         // if (cm_local_id(2) == 0 && cm_group_id(2) == 0) {
//         //    show(dscale.format<half, 16, 1>());
//         // }
//         //# St = k @ Qt

//         matrix<float, kv_step, q_step> St; // = ugemm_KQ(slm_K, rQ, slm_offset);
//         {
//             constexpr int num_K = kv_step/REG_M;
//             auto St2 = St.format<float, num_K, REG_M*REG_N>();
//             matrix<half, num_K, REG_M * REG_K> Kmat;
//             auto quan_Kmat = Kmat.format<uint8_t, 2, num_K*REG_M*REG_K>().row(1).format<uint8_t, REG_M*num_K, REG_K>();
//             auto dq_Kmat =  Kmat.format<half, REG_M*num_K, REG_K>();
//             //cm_slm_block_read(slm_K, GENX_NONE, slm_offset, Kmat.format<half>());

//             b2dKV_prefetch.set_base_ptr(reinterpret_cast<uint8_t*>(k_cache_base+prefetch_block_id*quan_blk_stride));
//             b2dKV_prefetch.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);
//             cm_prefetch<CacheHint::Cached, CacheHint::Cached>(b2dKV_prefetch.set_block_x(0));

//             b2dKV.set_base_ptr(reinterpret_cast<uint8_t*>(k_cache_base+cur_block_id*quan_blk_stride));
//             b2dKV.set_block_y(kv_pos%CMPA_BLOCK_SZ);

//             cm_load<lsc::Normal>(quan_Kmat.format<uint8_t>(), b2dKV.set_block_x(0));
//             // if (cm_local_id(2) == 0 && cm_group_id(2) == 0) {
//             //     show(quan_Kmat.format<uint8_t, 16, 16>(), false);
//             // }
//             #pragma unroll
//             for(int r = 0; r < kv_step; r++)  {
//                 dq_Kmat[r] =  quan_Kmat[r] - zp[r];
//                 dq_Kmat[r] = cm_mul<half>(dq_Kmat[r], dscale[r]);
//             }

//             #pragma unroll
//             for(int k = 0; k < num_K; k++)
//                 St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
//                                 0,
//                                 rQ[0].format<int32_t>(),
//                                 Kmat[k].format<int32_t>());

//             #pragma unroll
//             for(int ri = 1; ri < head_size/REG_K; ri++) {
//                 cm_prefetch<CacheHint::Cached, CacheHint::Cached>(b2dKV_prefetch.set_block_x(ri*REG_K));
//                 //cm_load<lsc::Normal>(Kmat.format<half>(), b2dKV.set_block_x(ri*REG_K));
//                 cm_load<lsc::Normal>(quan_Kmat.format<uint8_t>(), b2dKV.set_block_x(ri*REG_K));
//                 #pragma unroll
//                 for(int r = 0; r < kv_step; r++)  {
//                     dq_Kmat[r] =  quan_Kmat[r] - zp[r];
//                     dq_Kmat[r] = cm_mul<half>(dq_Kmat[r], dscale[r]);
//                 }
//                 #pragma unroll
//                 for(int k = 0; k < num_K; k++) {
//                     St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
//                         St2.row(k),
//                         rQ[ri].format<int32_t>(),
//                         Kmat[k].format<int32_t>());
//                 }
//             }
//         }
//         if constexpr (use_causal_mask) {
//             // since kv_step == q_step == 16, causal_left is n*kv_step
//             if (causal_left == 0) {
//                 apply_causal_mask<1>(St);
//             } else if (causal_left < 0) {
//                 St = -3.4e38f;
//             }
//             causal_left -= kv_step;
//         } else {
//             int kv_tokens = kv_stop - kv_pos;
//             // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
//             for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
//         }

//         //show(St);
//         auto max_comp = online_softmax_update(St, cur_max, cur_sum);

//         matrix<half, REG_N, REG_K> P;
//         Transpose2DMatrix(St, P);

//         b2dKV_prefetch.set_base_ptr(reinterpret_cast<uint8_t*>(v_cache_base+prefetch_block_id*quan_blk_stride));
//         b2dKV_prefetch.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);

//         b2dKV.set_base_ptr(reinterpret_cast<uint8_t*>(v_cache_base+cur_block_id*quan_blk_stride));
//         b2dKV.set_block_y(kv_pos%CMPA_BLOCK_SZ);


//         cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base+dscale_offset), dscale);
//         cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base+dscale_offset+CMPA_BLOCK_SZ*sizeof(half)), zp);

//         {
//             matrix<half, REG_K/2, REG_N*2> VmatVNNI2;
//             matrix<half, REG_K, REG_N> Vmat;
//             auto quanVmat = Vmat.format<half, 2, REG_K*REG_N/2>().row(1).format<uint8_t, REG_K, REG_N>();
//             int kv_tokens = kv_stop - kv_pos;
//             if (kv_pos == 0) {
//                 // ugemm_PV0(slm_V, P, rO, slm_offset);
//                 auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
//                 #pragma unroll
//                 for(int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
//                     cm_prefetch<CacheHint::Cached, CacheHint::Cached>(b2dKV_prefetch.set_block_x(k));
//                     cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dKV.set_block_x(k));
//                      #pragma unroll
//                     for(int r = 0; r < kv_step;r++)  {
//                         Vmat[r] =  quanVmat[r]-zp[r];
//                         Vmat[r] = cm_mul<half>(Vmat[r], dscale[r]);
//                     }
//                     for(int r = kv_step-1; r>=kv_tokens;r--)  {
//                         Vmat[r] = 0;
//                     }

//                     prepackAsVNNIWidth2(Vmat, VmatVNNI2);

//                     #pragma unroll
//                     for(int p = 0; p < num_P_tiles; p++) {
//                         rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
//                                         0,
//                                         VmatVNNI2.format<int32_t>(),
//                                         P2.row(p).format<int32_t>());
//                     }
//                 }
//             }
//             else {
//                 //ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
//                 auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
//                 #pragma unroll
//                 for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
//                     cm_prefetch<CacheHint::Cached, CacheHint::Cached>(b2dKV_prefetch.set_block_x(k));
//                     cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dKV.set_block_x(k));

//                     #pragma unroll
//                     for(int r = 0; r < kv_step;r++)  {
//                         Vmat[r] =  quanVmat[r]-zp[r];
//                         Vmat[r] = cm_mul<half>(Vmat[r], dscale[r]);
//                     }
//                     for(int r = kv_step-1; r>=kv_tokens;r--)  {
//                         Vmat[r] = 0;
//                     }

//                     prepackAsVNNIWidth2(Vmat, VmatVNNI2);
//                     //# compensate cur_O
//                     //  matrix <float, head_size/REG_K*2, REG_M*REG_N> rO;
//                     #pragma unroll
//                     for(int p = 0; p < num_P_tiles; p++) {
//                         auto cO = rO[ri + p].format<float, REG_M, REG_N>();
//                         #pragma unroll
//                         for(int r = 0; r < REG_M; r++)
//                             cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
//                     }

//                     #pragma unroll
//                     for(int p = 0; p < num_P_tiles; p++) {
//                         rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
//                                     rO[ri + p].format<float>(),
//                                     VmatVNNI2.format<int32_t>(),
//                                     P2.row(p).format<int32_t>());
//                     }
//                 }
//             }
//         }
//     }
//     if (q_tokens_left == 0) return;

//     //# save cur_O/cur_sum.transpose(0, 1)
//     matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
//     cur_sum = cm_inv(cur_sum);

//     lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_left - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);

//     #pragma unroll
//     for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
//         #pragma unroll
//         for(int p = 0; p < num_P_tiles; p++) {
//             auto cO = rO[ri + p].format<float, REG_M, REG_N>();
//             #pragma unroll
//             for(int r = 0; r < cO.n_rows(); r++) {
//                 cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);

//             }
//         }
//         b2dO.set_block_x(k);
//         cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
//         cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
//     }
// }

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
    // round up head_size to multiple of 16
    // block_2d_desc will automatically handle the tailing block
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;
    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, padded_head_size/REG_K, REG_K*REG_N> rQ;
    matrix <float, padded_head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    rO = 0.0f;  // Zero the accumulator: the first softmax block scales rO by max_comp==0, and 0*NaN==NaN if the GRF holds stale NaN/Inf bits.

    auto q_tokens_left = q_len;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < padded_head_size/2; k += REG_K/2, ri++) {
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
                for(int k = REG_K*wg_local_id; k < padded_head_size; k += REG_K*(local_size/2)) {
                    cm_load<lsc::Normal>(temp0, b2dK.set_block_x(k));
                    cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), temp0);
                }
            } else {
                vector<half, REG_K*REG_N> temp2;
                b2dV.set_block_y(kv_pos);
                #pragma unroll
                for(int k = REG_N*(wg_local_id-(local_size/2)); k < padded_head_size; k += REG_N*(local_size/2)) {
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
    for(int k = 0, ri=0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
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
    // round up head_size to multiple of 16
    // block_2d_desc will automatically handle the tailing block
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, padded_head_size/REG_K, REG_K*REG_N> rQ;
    matrix <float, padded_head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    rO = 0.0f;

    auto q_tokens_left = q_len;// - q_start;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    if (q_tokens_left > 0) {
        // Fold log2(e) into the Q pre-scale so St = K@Q^T lands in the log2 domain; the
        // online softmax then uses cm_exp (== exp2) directly, dropping a *log2e per St
        // element (16 muls/tile off the softmax critical path). Math is identical.
        constexpr float qscale = scale_factor * 1.4426950408889634f;
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_left - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < padded_head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)qscale);
        }
    }

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    int causal_left = q_start;

    // KV-blocking factor: number of kv tiles (each kv_step rows) processed per online-
    // softmax update. The per-iteration rO rescale (REG_M*num_rO_tiles muls) and the
    // softmax bookkeeping amortize over KV_BLK tiles, cutting the ALU-bound consume cost.
    // The K@Q^T and the St->P transpose are still per sub-tile (they don't amortize).
    constexpr int KV_BLK = CMFLA_KV_BLK;
    constexpr int num_K = kv_step/REG_M;
    constexpr int BLK_ROWS = KV_BLK*kv_step;

    // Warm up first K and V tiles before the loop so kv_pos=0 finds them in cache.
    prefetch_K.set_block_y(wg_local_id);
    prefetch_V.set_block_y(wg_local_id);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));
    #pragma unroll
    for(int ri = 1; ri < padded_head_size/REG_K; ri++) {
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri*REG_K));
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri*REG_N));
    }

    for(int kv_base = 0; kv_base < kv_stop; kv_base += BLK_ROWS) {
        //# St = K @ Q^T for KV_BLK stacked sub-tiles -> [KV_BLK*kv_step, q_step]
        matrix<float, BLK_ROWS, q_step> St;
        auto St2 = St.format<float, KV_BLK*num_K, REG_M*REG_N>();
        #pragma unroll
        for(int b = 0; b < KV_BLK; b++) {
            int kv_pos = kv_base + b*kv_step;
            matrix<half, num_K, REG_M * REG_K> Kmat;
            // Prefetch K of the matching tile one block ahead and V of this tile, so V
            // is warm by the PV phase and the next block's K is warm by its K phase.
            prefetch_K.set_block_y(wg_local_id + kv_pos + BLK_ROWS);
            prefetch_V.set_block_y(wg_local_id + kv_pos);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));

            b2dK.set_block_y(kv_pos);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            #pragma unroll
            for(int k = 0; k < num_K; k++)
                St2.row(b*num_K + k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0,
                                rQ[0].format<int32_t>(),
                                Kmat[k].format<int32_t>());

            #pragma unroll
            for(int ri = 1; ri < padded_head_size/REG_K; ri++) {
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri*REG_K));
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri*REG_N));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(ri*REG_K));
                #pragma unroll
                for(int k = 0; k < num_K; k++) {
                    St2.row(b*num_K + k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St2.row(b*num_K + k),
                        rQ[ri].format<int32_t>(),
                        Kmat[k].format<int32_t>());
                }
            }
        }

        // ---- mask per sub-tile (causal or kv-tail) ----
        if constexpr (use_causal_mask) {
            #pragma unroll
            for(int b = 0; b < KV_BLK; b++) {
                auto Stb = St.select<kv_step, 1, q_step, 1>(b*kv_step, 0);
                int cl = causal_left - b*kv_step;
                if (cl == 0) {
                    apply_causal_mask<1>(Stb);
                } else if (cl < 0) {
                    Stb = -3.4e38f;
                }
            }
            causal_left -= BLK_ROWS;
        } else {
            int kv_tokens = kv_stop - kv_base;
            // LSC ensures no overflow-access, but mask off k-tail attn-score is still required
            for(int p = kv_tokens; p < BLK_ROWS; p++) St[p] = -3.4e38f;
        }

        // ---- one online-softmax update over the whole block ----
        auto max_comp = online_softmax_update_tree(St, cur_max, cur_sum);

        // ---- rescale rO (skip first iter only when KV blocking amortizes the branch) ----
        // The kv_base=0 rescale is mathematically redundant for every head size: max_comp=0
        // and rO is still zero.  The skip, however, adds a runtime kv_base branch to every
        // outer KV block.  Enable it only for KV_BLK>=2, where fewer outer iterations make
        // the branch cost small enough to be profitable; KV_BLK=1 paths such as Omni HD=72
        // keep the straight-line rescale because forcing the branch measured as a regression.
        if constexpr (KV_BLK >= 2) {
            if (kv_base > 0) {
                #pragma unroll
                for(int t = 0; t < padded_head_size/REG_N*num_P_tiles; t++) {
                    auto cO = rO[t].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + (t % num_P_tiles)*REG_M]);
                }
            }
        } else {
            #pragma unroll
            for(int t = 0; t < padded_head_size/REG_N*num_P_tiles; t++) {
                auto cO = rO[t].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < REG_M; r++)
                    cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + (t % num_P_tiles)*REG_M]);
            }
        }

        // ---- transpose each sub-tile and accumulate P@V into rO ----
        constexpr int num_Vchunks = padded_head_size/REG_N;
        #pragma unroll
        for(int b = 0; b < KV_BLK; b++) {
            matrix<half, REG_N, REG_K> P;
            transpose_St_to_P_half(St.select<kv_step, 1, q_step, 1>(b*kv_step, 0), P);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            b2dV.set_block_y(kv_base + b*kv_step);
            #pragma unroll
            for(int ci = 0, ri = 0; ci < num_Vchunks; ci++, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(ci*REG_N));
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
    for(int k = 0, ri=0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
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

// Item 9: process 2 Q-rows per thread — amortize per-kv softmax cost over 2x DPAS work.
// q_len is the number of Q tokens for the FIRST row; the second row is at q_base + q_pitch.
// The caller must ensure at most 2*q_step tokens are dispatched to this thread.
template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
void sdpa_kernel_lsc_prefetch_q2(
    int wg_local_id,
    int q_start,
    int kv_stop,
    int q_len,         // tokens for this thread, up to 2*q_step
    int kv_len,
    svmptr_t q_base [[type("svmptr_t")]],
    svmptr_t k_base [[type("svmptr_t")]],
    svmptr_t v_base [[type("svmptr_t")]],
    svmptr_t o_base [[type("svmptr_t")]]) {

    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads*2) * head_size * sizeof(half)) : o_pitch;
    constexpr uint kv_pitch = is_qkv_fused ? q_pitch : (num_kv_heads * head_size * sizeof(half));
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 * 16;

    // Two independent softmax states for the two Q rows.
    vector<float, q_step> cur_max_a, cur_sum_a;
    vector<float, q_step> cur_max_b, cur_sum_b;
    cur_max_a = -3e38f; cur_sum_a = 0;
    cur_max_b = -3e38f; cur_sum_b = 0;

    constexpr int num_P_tiles = REG_N / REG_M;
    // Two rQ halves: rQ_a for first row, rQ_b for second row
    matrix<half, padded_head_size/REG_K, REG_K*REG_N> rQ_a;
    matrix<half, padded_head_size/REG_K, REG_K*REG_N> rQ_b;
    // Two rO halves
    matrix<float, padded_head_size/REG_N*num_P_tiles, REG_M*REG_N> rO_a;
    matrix<float, padded_head_size/REG_N*num_P_tiles, REG_M*REG_N> rO_b;
    rO_a = 0.0f;  // See rO note above: avoid 0*NaN from stale GRF bits.
    rO_b = 0.0f;  // See rO note above: avoid 0*NaN from stale GRF bits.

    int q_len_a = q_len;
    if (q_len_a < 0) q_len_a = 0;
    if (q_len_a > q_step) q_len_a = q_step;

    int q_len_b = q_len - q_step;
    if (q_len_b < 0) q_len_b = 0;
    if (q_len_b > q_step) q_len_b = q_step;

    // Load rQ_a (first Q row)
    if (q_len_a > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQa(reinterpret_cast<uint*>(q_base), q_len_a - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < padded_head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ_a[ri].format<uint>(), b2dQa.set_block_x(k));
            rQ_a[ri].format<half>() = cm_mul<half>(rQ_a[ri].format<half>(), (half)scale_factor);
        }
    }
    // Load rQ_b (second Q row block: q_step tokens further = q_step * q_pitch bytes)
    if (q_len_b > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQb(reinterpret_cast<uint*>(q_base + (svmptr_t)q_step * q_pitch), q_len_b - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < padded_head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ_b[ri].format<uint>(), b2dQb.set_block_x(k));
            rQ_b[ri].format<half>() = cm_mul<half>(rQ_b[ri].format<half>(), (half)scale_factor);
        }
    }

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_base, kv_stop - 1, head_size*sizeof(half) - 1, kv_pitch - 1, 0, 0);

    int causal_left = q_start;

    // Warm-up prefetch for kv_pos=0 K+V
    prefetch_K.set_block_y(wg_local_id);
    prefetch_V.set_block_y(wg_local_id);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));
    #pragma unroll
    for(int ri = 1; ri < padded_head_size/REG_K; ri++) {
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri*REG_K));
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri*REG_N));
    }

    for(int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step,
            k_base += kv_step * kv_pitch,
            v_base += kv_step * kv_pitch) {

        // K phase: compute St_a = K × rQ_a and St_b = K × rQ_b
        matrix<float, kv_step, q_step> St_a;
        matrix<float, kv_step, q_step> St_b;
        {
            constexpr int num_K = kv_step/REG_M;
            auto St_a2 = St_a.format<float, num_K, REG_M*REG_N>();
            auto St_b2 = St_b.format<float, num_K, REG_M*REG_N>();

            matrix<half, num_K, REG_M * REG_K> Kmat;

            prefetch_K.set_block_y(wg_local_id + kv_pos + kv_step);
            prefetch_V.set_block_y(wg_local_id + kv_pos);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(0));
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(0));

            b2dK.set_block_y(kv_pos);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            #pragma unroll
            for(int k = 0; k < num_K; k++) {
                St_a2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0, rQ_a[0].format<int32_t>(), Kmat[k].format<int32_t>());
                St_b2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0, rQ_b[0].format<int32_t>(), Kmat[k].format<int32_t>());
            }

            #pragma unroll
            for(int ri = 1; ri < padded_head_size/REG_K; ri++) {
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(ri*REG_K));
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(ri*REG_N));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(ri*REG_K));
                #pragma unroll
                for(int k = 0; k < num_K; k++) {
                    St_a2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St_a2.row(k), rQ_a[ri].format<int32_t>(), Kmat[k].format<int32_t>());
                    St_b2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St_b2.row(k), rQ_b[ri].format<int32_t>(), Kmat[k].format<int32_t>());
                }
            }
        }

        // Tail masking
        {
            int kv_tokens = kv_stop - kv_pos;
            for(int p = kv_tokens; p < kv_step; p++) {
                St_a[p] = -3.4e38f;
                St_b[p] = -3.4e38f;
            }
        }

        auto max_comp_a = online_softmax_update(St_a, cur_max_a, cur_sum_a);
        auto max_comp_b = online_softmax_update(St_b, cur_max_b, cur_sum_b);

        matrix<half, REG_N, REG_K> P_a, P_b;
        Transpose2DMatrix(St_a, P_a);
        Transpose2DMatrix(St_b, P_b);

        // V phase: unified path (max_comp=0 at kv_pos=0 zeroes rO)
        b2dV.set_block_y(kv_pos);
        {
            auto Pa2 = P_a.format<half, num_P_tiles, REG_M * REG_K>();
            auto Pb2 = P_b.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri = 0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
                // Rescale and DPAS for rO_a
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    auto cO = rO_a[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp_a[r + p*REG_M]);
                }
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO_a[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_a[ri + p].format<float>(), Vmat.format<int32_t>(), Pa2.row(p).format<int32_t>());
                }
                // Rescale and DPAS for rO_b
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    auto cO = rO_b[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp_b[r + p*REG_M]);
                }
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO_b[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_b[ri + p].format<float>(), Vmat.format<int32_t>(), Pb2.row(p).format<int32_t>());
                }
            }
        }
    }

    // Store output for first Q row
    if (q_len_a > 0) {
        matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
        cur_sum_a = cm_inv(cur_sum_a);
        lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_len_a - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri=0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
            #pragma unroll
            for(int p = 0; p < num_P_tiles; p++) {
                auto cO = rO_a[ri + p].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < cO.n_rows(); r++)
                    cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum_a[r + p*REG_M]);
            }
            b2dO.set_block_x(k);
            cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
            cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
        }
    }

    // Store output for second Q row
    if (q_len_b > 0) {
        matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
        cur_sum_b = cm_inv(cur_sum_b);
        lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base + (svmptr_t)q_step * o_pitch, q_len_b - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri=0; k < padded_head_size; k += REG_N, ri += num_P_tiles) {
            #pragma unroll
            for(int p = 0; p < num_P_tiles; p++) {
                auto cO = rO_b[ri + p].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < cO.n_rows(); r++)
                    cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum_b[r + p*REG_M]);
            }
            b2dO.set_block_x(k);
            cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
            cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
        }
    }
}

#else  // CM_HAS_LSC_UNTYPED_2D

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
    constexpr int padded_head_size = (head_size + 16 - 1) / 16 *16;
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
            constexpr int num_full_blocks = head_size / REG_K;
            int i = 0;
            #pragma unroll
            for(; i < num_full_blocks; i++) {
                matrix<uint, q_step, REG_K/2> QmatI32;
                int k = i * REG_K;
                cm_load_2d(QmatI32, query, q_off + k * sizeof(uint) / 2, q_pitch);
                Transpose2DMatrix(QmatI32, rQ[i].format<uint, REG_K/2, q_step>());
                rQ[i].format<half>() = cm_mul<half>(rQ[i].format<half>(), (half)scale_factor);
            }

            // if with tail, load with head_size_tail
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % REG_K > 0) {
                int k = num_full_blocks * REG_K;
                matrix<uint, q_step, REG_K/2> QmatI32;
                cm_load_2d_with_tail<q_step, REG_K/2, (head_size % REG_K) / 2>(QmatI32, query, q_off + k * sizeof(half), q_pitch);
                Transpose2DMatrix(QmatI32, rQ[num_full_blocks].format<uint, REG_K/2, q_step>());
                rQ[num_full_blocks].format<half>() = cm_mul<half>(rQ[num_full_blocks].format<half>(), (half)scale_factor);
            }
        }
    }

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix <float, padded_head_size/REG_N*num_P_tiles, REG_M*REG_N> rO;
    rO = 0.0f;    // Zero the accumulator: the first softmax block scales rO by max_comp==0, and 0*NaN==NaN if the GRF holds stale NaN/Inf bits.
    int causal_left = q_start;

    constexpr uint slm_buff_size = kv_step * padded_head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read = 0;

    auto load_slm_KV = [&](int kv_pos) {
        //if (kv_pos < 1024000) return;
        int kv_tokens = kv_stop - kv_pos;
        if (kv_tokens <= 0) return;
        // Calculate valid rows for this block (used to zero out garbage data)
        int kv_valid_rows = (kv_tokens >= kv_step) ? kv_step : kv_tokens;
        uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
        slm_buff_id_write ++;

        // non-tail branch is faster
        if (wg_local_id < local_size/2) {
            //if (kv_pos > 1024000) {
            matrix<half, 2*REG_M, REG_K> temp;
            // head_size is split into blocks of <half, REG_K>
            constexpr int num_full_blocks = head_size / REG_K;
            int i = wg_local_id;
            for (; i < num_full_blocks; i += local_size / 2) {
                int k = i * REG_K;
                cm_load_2d(temp, key, k_off + k * sizeof(half), kv_pitch);
                // Zero out unused K rows to prevent NaN from garbage data in KV cache
                // (Similar approach as PA kernel: cm_pa_common.hpp)
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp.row(r) = 0;
                cm_slm_block_write(slm_K,
                    slm_offset + k * 2 * REG_M * sizeof(half),
                    temp.format<half>());
            }
            // if with tail, load with head_size % REG_K
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % REG_K > 0) {
                int k = num_full_blocks * REG_K;
                cm_load_2d_with_tail<2*REG_M, REG_K, head_size % REG_K>(temp, key, k_off + k * sizeof(half), kv_pitch);
                // Zero out unused K rows
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp.row(r) = 0;
                cm_slm_block_write(slm_K,
                    slm_offset + k * 2 * REG_M * sizeof(half),
                    temp.format<half>());
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
            constexpr int num_full_blocks = head_size / VK_STEP;
            int i = wg_local_id - local_size / 2;
            for (; i < num_full_blocks; i += local_size / 2) {
                int k = i * VK_STEP;
                cm_load_2d(temp2, value, v_off + k * sizeof(half), kv_pitch);
                // Zero out unused V rows to prevent NaN from garbage data in KV cache
                // (Similar approach as PA kernel: cm_pa_common.hpp)
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp2.row(r) = 0;
                #pragma unroll
                for (int p = 0; p < VK_STEP / REG_N; p++) {
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K / 2, 2, REG_N, 1>(0, p * REG_N);
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K / 2, 2, REG_N, 1>(1, p * REG_N);

                    cm_slm_block_write(slm_V, slm_offset + (k + p * REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
                }
            }
            // if with tail, load with head_size_tail
            // following code will be optimized out when head_size_tail = 0
            if constexpr (head_size % VK_STEP > 0) {
                int k = num_full_blocks * VK_STEP;
                cm_load_2d_with_tail<REG_K, VK_STEP, head_size % VK_STEP>(temp2, value, v_off + k * sizeof(half), kv_pitch);
                // Zero out unused V rows
                for (int r = kv_valid_rows; r < kv_step; r++)
                    temp2.row(r) = 0;
                #pragma unroll
                for (int p = 0; p < VK_STEP / REG_N; p++) {
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 0) = temp2.select<REG_K / 2, 2, REG_N, 1>(0, p * REG_N);
                    temp_vnni.select<REG_K / 2, 1, REG_N, 2>(0, 1) = temp2.select<REG_K / 2, 2, REG_N, 1>(1, p * REG_N);

                    cm_slm_block_write(slm_V, slm_offset + (k + p * REG_N) * REG_K * sizeof(half), temp_vnni.format<half>());
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

#endif  // !CM_HAS_LSC_UNTYPED_2D