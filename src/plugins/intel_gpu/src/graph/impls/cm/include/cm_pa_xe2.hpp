// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CM_HAS_LSC_UNTYPED_2D

#ifndef CMPA_WG_SEQ_LEN
#error "CMPA_WG_SEQ_LEN must be defined"
#endif

#ifndef SPARSE_BLOCK_SIZE
#error "SPARSE_BLOCK_SIZE must be defined"
#endif

#ifndef KV_CACHE_COMPRESSION
#error "KV_CACHE_COMPRESSION must be defined"
#endif

#if KV_CACHE_COMPRESSION
#define OPTIMIZED_SPARSE_PIPELINE ( \
                                   (SPARSE_BLOCK_SIZE == CMPA_WG_SEQ_LEN) && \
                                   (SPARSE_BLOCK_SIZE == 128 || SPARSE_BLOCK_SIZE == 256) && \
                                   ((CMPA_BLOCK_SZ % SPARSE_BLOCK_SIZE) == 0) \
                                  )
#else
#define OPTIMIZED_SPARSE_PIPELINE ( \
                                   (SPARSE_BLOCK_SIZE == CMPA_WG_SEQ_LEN) && \
                                   (SPARSE_BLOCK_SIZE == 128 || SPARSE_BLOCK_SIZE == 256) \
                                  )
#endif

#if OPTIMIZED_SPARSE_PIPELINE == 1
    static_assert(SPARSE_BLOCK_SIZE == CMPA_WG_SEQ_LEN,
        "This optimized pipeline requires SPARSE_BLOCK_SIZE to be the same as CMPA_WG_SEQ_LEN");
    static_assert(SPARSE_BLOCK_SIZE == 128 || SPARSE_BLOCK_SIZE == 256,
        "This optimized pipeline assumes SPARSE_BLOCK_SIZE is 128 or 256 to efficiently index sparse blocks");
#if KV_CACHE_COMPRESSION
    static_assert((CMPA_BLOCK_SZ % SPARSE_BLOCK_SIZE) == 0,
        "U8 optimized sparse pipeline assumes CMPA_BLOCK_SZ is divisible by SPARSE_BLOCK_SIZE to efficiently index kvcache blocks");
#endif
#endif

#if KV_CACHE_COMPRESSION
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
    constexpr uint q_pitch = is_q_fused
        ? ((num_heads + num_kv_heads * 2) * head_size * sizeof(half))
        : o_pitch;

    constexpr uint kv_pitch = head_size * sizeof(uint8_t);

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, head_size / REG_K, REG_K * REG_N> rQ;
    matrix<float, head_size / REG_N * num_P_tiles, REG_M * REG_N> rO;
    bool first_active = true;

    auto q_tokens_left = q_len;
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);
#if KV_CACHE_COMPRESSION == 2
    static_assert(SUB_BLOCK_SIZE % 16 == 0, "SUB_BLOCK_SIZE must be divisible by 16");
    static_assert(CMPA_BLOCK_SZ % SUB_BLOCK_SIZE == 0, "CMPA_BLOCK_SZ must be divisible by SUB_BLOCK_SIZE");
#endif

    if (q_tokens_left < 0) q_tokens_left = 0;
    if (q_tokens_left > q_step) q_tokens_left = q_step;

    // ---- Load Q (unchanged) ----
    if (q_tokens_left > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K / 2> b2dQ(
            reinterpret_cast<uint*>(q_base),
            q_tokens_left - 1,
            head_size * sizeof(half) - 1,
            q_pitch - 1,
            0, 0);

        #pragma unroll
        for (int k = 0, ri = 0; k < head_size / 2; k += REG_K / 2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    }

    lsc::block_2d_desc<uint8_t, 1, kv_step, REG_K> b2dK(
        k_cache_base, CMPA_BLOCK_SZ - 1, head_size * sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<uint8_t, 1, REG_K, REG_N> b2dV(
        v_cache_base, CMPA_BLOCK_SZ - 1, head_size * sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);

#if KV_CACHE_COMPRESSION == 1
    constexpr int k_quan_blk_stride = CMFLA_NUM_KV_HEADS * (CMFLA_HEAD_SIZE + 4) * CMPA_BLOCK_SZ * sizeof(uint8_t);
#else
    constexpr int k_quan_blk_stride = CMFLA_NUM_KV_HEADS * CMFLA_HEAD_SIZE * (CMPA_BLOCK_SZ + CMPA_BLOCK_SZ / SUB_BLOCK_SIZE * 4) * sizeof(uint8_t);
#endif
    constexpr int v_quan_blk_stride = CMFLA_NUM_KV_HEADS * (CMFLA_HEAD_SIZE + 4) * CMPA_BLOCK_SZ * sizeof(uint8_t);

    int causal_left = q_start + past_lens;

    constexpr uint slm_buff_size = kv_step * head_size * sizeof(half);
    int slm_buff_id_write = 0;
    int slm_buff_id_read  = 0;

#if OPTIMIZED_SPARSE_PIPELINE == 1
    // ==================================================================================
    // Optimized block-granular sparse pipeline when SPARSE_BLOCK_SIZE == WG_SEQ_LEN
    // ==================================================================================
    constexpr int cmpa_shift = (CMPA_BLOCK_SZ == 256) ? 8 : (CMPA_BLOCK_SZ == 128) ? 7 : -1;
    constexpr int cmpa_mask = CMPA_BLOCK_SZ - 1;
    constexpr int sb_shift = (SPARSE_BLOCK_SIZE == 128) ? 7 : (SPARSE_BLOCK_SIZE == 256) ? 8 : -1;

    auto skip_by = [&](const bool* base, int kv_pos) -> bool {
        if constexpr (sb_shift < 0) {
            return false;
        } else {
            return !base[(uint)kv_pos >> sb_shift];
        }
    };

    auto get_mask_base = [&](bool for_load) -> const bool* {
        (void)for_load;
        return (const bool*)sparse_mask_base;
    };

    auto skip_load = [&](int kv_pos) -> bool {
        return skip_by(get_mask_base(true), kv_pos);
    };

    auto skip_compute = [&](int kv_pos) -> bool {
        return skip_by(get_mask_base(false), kv_pos);
    };

    // ============================================================
    // Maskless SLM loader for ACTIVE blocks (per-step skip removed)
    // ============================================================
    auto load_slm_KV_active = [&](int kv_pos, int blk_end, int kv_pos_in_block, int cur_block_id) {

        // Only load KV that will be consumed inside this active block.
        if (kv_pos >= blk_end) return;
        if (kv_pos >= kv_stop) return;

        // Ring slot for this load
        uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;

        // kv_left for tail within kv_stop
        int kv_left = kv_step;
        if (kv_pos + kv_step > kv_stop) kv_left = kv_stop - kv_pos;

#if KV_CACHE_COMPRESSION == 1
        uint32_t k_dscale_offset =
            cur_block_id * k_quan_blk_stride +
            CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) +
            kv_pos_in_block * sizeof(half);
        uint32_t k_zp_offset = k_dscale_offset + CMPA_BLOCK_SZ * sizeof(half);
#else
        uint32_t k_dscale_offset =
            cur_block_id * k_quan_blk_stride +
            CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) +
            kv_pos_in_block / SUB_BLOCK_SIZE * head_size * sizeof(half);
        uint32_t k_zp_offset = k_dscale_offset + CMPA_BLOCK_SZ / SUB_BLOCK_SIZE * head_size * sizeof(half);
#endif
        uint32_t v_dscale_offset =
            cur_block_id * v_quan_blk_stride +
            CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) +
            kv_pos_in_block * sizeof(half);
        uint32_t v_zp_offset = v_dscale_offset + CMPA_BLOCK_SZ * sizeof(half);

        // Advance write id exactly when we actually stage data
        slm_buff_id_write++;

        vector<half, kv_step> dscale;
        vector<half, kv_step> zp;

        if (wg_local_id < local_size / 2) {
            // ---- Load K scales / zps ----
#if KV_CACHE_COMPRESSION == 1
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_dscale_offset), dscale);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_zp_offset), zp);
#endif

            matrix<half, kv_step, REG_K> kmat;
            auto quanKmat =
                kmat.format<half, 2, kv_step * REG_K / 2>()[1].format<uint8_t, kv_step, REG_K>();

            b2dK.set_base_ptr(reinterpret_cast<uint8_t*>(k_cache_base + cur_block_id * k_quan_blk_stride));
            b2dK.set_block_y(kv_pos_in_block);

            for (int k = REG_K * wg_local_id; k < head_size; k += REG_K * (local_size / 2)) {
#if KV_CACHE_COMPRESSION == 2
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_dscale_offset + k * sizeof(half)), dscale);
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_zp_offset + k * sizeof(half)), zp);
#endif
                cm_load<lsc::Normal>(quanKmat.format<uint8_t>(), b2dK.set_block_x(k));

                #pragma unroll
                for (int r = 0; r < kv_step; r++) {
#if KV_CACHE_COMPRESSION == 1
                    kmat[r] = quanKmat[r] - zp[r];
                    kmat[r] = cm_mul<half>(kmat[r], dscale[r]);
#else
                    kmat[r] = quanKmat[r] - zp;
                    kmat[r] = cm_mul<half>(kmat[r], dscale);
#endif
                }

                if (kv_left < kv_step) {
                    for (int r = kv_step - 1; r >= kv_left; r--) kmat[r] = 0;
                }

                cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), kmat.format<half>());
            }
        } else {
            // ---- Load V scales / zps ----
            cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + v_dscale_offset), dscale);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + v_zp_offset), zp);

            matrix<half, REG_K / 2, REG_N * 2> VmatVNNI;
            matrix<half, REG_K, REG_N> Vmat;
            auto quanVmat =
                Vmat.format<half, 2, REG_K * REG_N / 2>().row(1).format<uint8_t, REG_K, REG_N>();

            b2dV.set_base_ptr(reinterpret_cast<uint8_t*>(v_cache_base + cur_block_id * v_quan_blk_stride));
            b2dV.set_block_y(kv_pos_in_block);

            #pragma unroll
            for (int k = REG_N * (wg_local_id - (local_size / 2));
                 k < head_size;
                 k += REG_N * (local_size / 2)) {

                cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(k));

                #pragma unroll
                for (int r = 0; r < kv_step; r++) {
                    Vmat[r] = quanVmat[r] - zp[r];
                    Vmat[r] = cm_mul<half>(Vmat[r], dscale[r]);
                }

                if (kv_left < kv_step) {
                    for (int r = kv_step - 1; r >= kv_left; r--) Vmat[r] = 0;
                }

                prepackAsVNNIWidth2(Vmat, VmatVNNI);
                cm_slm_block_write(slm_V, slm_offset + k * REG_K * sizeof(half), VmatVNNI.format<half>());
            }
        }
    };

    // ============================================================
    // Block-granular sparse gating + block-local pipeline
    // ============================================================
    constexpr int KV_BLOCK = SPARSE_BLOCK_SIZE;
    constexpr int STEPS_PER_BLOCK = KV_BLOCK / kv_step;

    for (int kv_blk = 0; kv_blk < kv_stop; kv_blk += KV_BLOCK) {

        int blk_end = kv_blk + KV_BLOCK;
        if (blk_end > kv_stop) blk_end = kv_stop;

        int blk_len = blk_end - kv_blk;
        if (blk_len <= 0) break;

        int steps_in_block = (blk_len + kv_step - 1) / kv_step; // <= 16

        // Per-block skip (mask constant within SPARSE_BLOCK_SIZE tokens)
        const bool block_sparse = skip_load(kv_blk);

        if (block_sparse) {
            if constexpr (use_causal_mask) {
                causal_left -= steps_in_block * kv_step;
            }
            continue;
        }

        {
            // Reset ring counters per active block
            slm_buff_id_write = 0;
            slm_buff_id_read  = 0;

            auto kv_blk_block_id = block_indices[kv_blk >> cmpa_shift];
            auto kv_blk_pos_in_block = kv_blk & cmpa_mask;

            // Prime pipeline (avoid 2nd prime if block too short)
            load_slm_KV_active(kv_blk, blk_end, kv_blk_pos_in_block, kv_blk_block_id);
            if (kv_blk + kv_step < blk_end)
                load_slm_KV_active(kv_blk + kv_step, blk_end, (kv_blk + kv_step) & cmpa_mask,
                    block_indices[(kv_blk + kv_step) >> cmpa_shift]);

            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_sbarrier(1);

            for (int kv_pos = kv_blk; kv_pos < blk_end; kv_pos += kv_step, slm_buff_id_read++) {

                cm_fence(CM_LOCAL_BARRIER);
                cm_sbarrier(0);

                // Prefetch 2 steps ahead only if it stays within this block
                if (kv_pos + 2 * kv_step < blk_end) {
                    int prefetch_kv_pos = kv_pos + 2 * kv_step;
                    load_slm_KV_active(prefetch_kv_pos, blk_end, prefetch_kv_pos & cmpa_mask,
                        block_indices[prefetch_kv_pos >> cmpa_shift]);
                    cm_slm_fence(CM_LOCAL_BARRIER);
                }

                if (kv_pos + kv_step < blk_end)
                    cm_sbarrier(1);

                {
                    uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;

                    matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

                    if constexpr (use_causal_mask) {
                        if (causal_left == 0) {
                            apply_causal_mask<1>(St);
                        } else if (causal_left < 0) {
                            St = -3.4e38f;
                        }
                        causal_left -= kv_step;
                    } else {
                        int kv_tokens = kv_stop - kv_pos;
                        for (int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
                    }

                    auto max_comp = online_softmax_update(St, cur_max, cur_sum);

                    matrix<half, REG_N, REG_K> P;
                    Transpose2DMatrix(St, P);

                    if (first_active) {
                        ugemm_PV0(slm_V, P, rO, slm_offset);
                        first_active = false;
                    } else {
                        ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
                    }
                }
            }
        }
    }

#else
    // ========================================================================
    // Legacy per-step pipeline for any SPARSE_BLOCK_SIZE (including 1)
    // ======================================================================
#if SPARSE_BLOCK_SIZE > 1
    constexpr int sb_shift = (SPARSE_BLOCK_SIZE == 128) ? 7 : (SPARSE_BLOCK_SIZE == 256) ? 8 : -1;
    auto skip_by = [&](const bool* base, int kv_pos) -> bool {
        if constexpr (sb_shift < 0) {
            return false;
        } else {
            return !base[(uint)kv_pos >> sb_shift];
        }
    };

    auto skip_compute = [&](int kv_pos) -> bool {
        return skip_by((const bool*)sparse_mask_base, kv_pos);
    };

    auto skip_load = [&](int kv_pos) -> bool {
        return skip_by((const bool*)wg_sparse_mask_base, kv_pos);
    };
#endif
    auto load_slm_KV = [&](int kv_pos) {
        if (kv_pos >= kv_stop) return;

#if SPARSE_BLOCK_SIZE > 1
        if (skip_load(kv_pos)) {
            slm_buff_id_write++;
            return;
        }
#endif
        int cur_block_id = block_indices[kv_pos / CMPA_BLOCK_SZ];
        int kv_pos_in_block = kv_pos - (kv_pos / CMPA_BLOCK_SZ) * CMPA_BLOCK_SZ;
#if KV_CACHE_COMPRESSION == 1
        uint32_t k_dscale_offset =
            cur_block_id * k_quan_blk_stride +
            CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) +
            kv_pos_in_block * sizeof(half);
        uint32_t k_zp_offset = k_dscale_offset + CMPA_BLOCK_SZ * sizeof(half);
#else
        uint32_t k_dscale_offset =
            cur_block_id * k_quan_blk_stride +
            CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) +
            kv_pos_in_block / SUB_BLOCK_SIZE * head_size * sizeof(half);
        uint32_t k_zp_offset = k_dscale_offset + CMPA_BLOCK_SZ / SUB_BLOCK_SIZE * head_size * sizeof(half);
#endif
        uint32_t v_dscale_offset =
            cur_block_id * v_quan_blk_stride +
            CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) +
            kv_pos_in_block * sizeof(half);
        uint32_t v_zp_offset = v_dscale_offset + CMPA_BLOCK_SZ * sizeof(half);

        uint slm_offset = (slm_buff_id_write & 3) * slm_buff_size;
        vector<half, kv_step> dscale;
        vector<half, kv_step> zp;
        int kv_left = (kv_stop - kv_pos) > kv_step ? kv_step : (kv_stop - kv_pos);

        slm_buff_id_write++;

        if (wg_local_id < local_size / 2) {
#if KV_CACHE_COMPRESSION == 1
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_dscale_offset), dscale);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_zp_offset), zp);
#endif

            matrix<half, kv_step, REG_K> kmat;
            auto quanKmat =
                kmat.format<half, 2, kv_step * REG_K / 2>()[1].format<uint8_t, kv_step, REG_K>();
            b2dK.set_base_ptr(reinterpret_cast<uint8_t*>(k_cache_base + cur_block_id * k_quan_blk_stride));
            b2dK.set_block_y(kv_pos_in_block);

            for (int k = REG_K * wg_local_id; k < head_size; k += REG_K * (local_size / 2)) {
#if KV_CACHE_COMPRESSION == 2
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_dscale_offset + k * sizeof(half)), dscale);
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_zp_offset + k * sizeof(half)), zp);
#endif
                cm_load<lsc::Normal>(quanKmat.format<uint8_t>(), b2dK.set_block_x(k));

                #pragma unroll
                for (int r = 0; r < kv_step; r++) {
#if KV_CACHE_COMPRESSION == 1
                    kmat[r] = quanKmat[r] - zp[r];
                    kmat[r] = cm_mul<half>(kmat[r], dscale[r]);
#else
                    kmat[r] = quanKmat[r] - zp;
                    kmat[r] = cm_mul<half>(kmat[r], dscale);
#endif
                }

                for (int r = kv_step - 1; r >= kv_left; r--) kmat[r] = 0;

                cm_slm_block_write(slm_K, slm_offset + k * kv_step * sizeof(half), kmat.format<half>());
            }
        } else {
            cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + v_dscale_offset), dscale);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + v_zp_offset), zp);

            matrix<half, REG_K / 2, REG_N * 2> VmatVNNI;
            matrix<half, REG_K, REG_N> Vmat;
            auto quanVmat =
                Vmat.format<half, 2, REG_K * REG_N / 2>().row(1).format<uint8_t, REG_K, REG_N>();
            b2dV.set_base_ptr(reinterpret_cast<uint8_t*>(v_cache_base + cur_block_id * v_quan_blk_stride));
            b2dV.set_block_y(kv_pos_in_block);

            #pragma unroll
            for (int k = REG_N * (wg_local_id - (local_size / 2));
                 k < head_size;
                 k += REG_N * (local_size / 2)) {

                cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(k));

                #pragma unroll
                for (int r = 0; r < kv_step; r++) {
                    Vmat[r] = quanVmat[r] - zp[r];
                    Vmat[r] = cm_mul<half>(Vmat[r], dscale[r]);
                }

                for (int r = kv_step - 1; r >= kv_left; r--) Vmat[r] = 0;

                prepackAsVNNIWidth2(Vmat, VmatVNNI);
                cm_slm_block_write(slm_V, slm_offset + k * REG_K * sizeof(half), VmatVNNI.format<half>());
            }
        }
    };

    load_slm_KV(0);
    load_slm_KV(kv_step);
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);

    for (int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step, slm_buff_id_read++) {
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

        if (kv_pos + kv_step < kv_stop)
            cm_sbarrier(1);
        load_slm_KV(kv_pos + kv_step * 2);

#if SPARSE_BLOCK_SIZE > 1
        if (skip_compute(kv_pos)) {
            if constexpr (use_causal_mask) {
                causal_left -= kv_step;
            }
            continue;
        }
#endif

        {
            uint slm_offset = (slm_buff_id_read & 3) * slm_buff_size;

            matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_offset);

            if constexpr (use_causal_mask) {
                if (causal_left == 0) {
                    apply_causal_mask<1>(St);
                } else if (causal_left < 0) {
                    St = -3.4e38f;
                }
                causal_left -= kv_step;
            } else {
                int kv_tokens = kv_stop - kv_pos;
                for (int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
            }

            auto max_comp = online_softmax_update(St, cur_max, cur_sum);

            matrix<half, REG_N, REG_K> P;
            Transpose2DMatrix(St, P);

            if (first_active) {
                ugemm_PV0(slm_V, P, rO, slm_offset);
                first_active = false;
            } else {
                ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
            }
        }
    }
#endif

    // ============================================================
    // Store O (unchanged)
    // ============================================================
    if (q_tokens_left == 0) return;

#ifdef CMPA_DEBUG_ALL_MASKED
    if (first_active) {
        cm_printf("CMPA error: all blocks masked out, q_start=%d\n", q_start);
    }
#endif

    matrix<half, num_P_tiles * REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(
        o_base,
        q_tokens_left - 1,
        head_size * sizeof(half) - 1,
        o_pitch - 1,
        0, 0);

    #pragma unroll
    for (int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {

        #pragma unroll
        for (int p = 0; p < num_P_tiles; p++) {
            auto cO = rO[ri + p].format<float, REG_M, REG_N>();

            #pragma unroll
            for (int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p * REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p * REG_M]);
            }
        }

        b2dO.set_block_x(k);
        cm_store(b2dO.set_block_y(0),
                 cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M),
                 cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
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
    bool first_active = true;

#if SPARSE_BLOCK_SIZE > 1
    constexpr int sb_shift = (SPARSE_BLOCK_SIZE == 128) ? 7 : (SPARSE_BLOCK_SIZE == 256) ? 8 : -1;
    auto skip_by = [&](const bool* base, int kv_pos) -> bool {
        if constexpr (sb_shift < 0) {
            return false;
        } else {
            return !base[(uint)kv_pos >> sb_shift];
        }
    };

    auto skip_compute = [&](int kv_pos) {
        return skip_by((const bool*)sparse_mask_base, kv_pos);
    };
#endif

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

    lsc::block_2d_desc<half, 1, kv_step, REG_K> b2dK(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dV(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<half, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, k_pitch - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(half) - 1, v_pitch - 1, 0, 0);
    constexpr int blk_stride = CMFLA_NUM_KV_HEADS*CMFLA_HEAD_SIZE*CMPA_BLOCK_SZ;
    int causal_left = q_start+past_lens;

#if OPTIMIZED_SPARSE_PIPELINE == 1
    // ====================================================================================
    // Optimized block-granular sparse pipeline when SPARSE_BLOCK_SIZE == WG_SEQ_LEN
    // ====================================================================================
    constexpr int kv_block = SPARSE_BLOCK_SIZE;
    for (int kv_blk = 0; kv_blk < kv_stop; kv_blk += kv_block) {
        int blk_end = kv_blk + kv_block;
        if (blk_end > kv_stop) blk_end = kv_stop;
        int blk_len = blk_end - kv_blk;
        if (blk_len <= 0) break;
        int steps_in_block = (blk_len + kv_step - 1) / kv_step;

        if (skip_compute(kv_blk)) {
            if constexpr (use_causal_mask) {
                causal_left -= steps_in_block * kv_step;
            }
            continue;
        }

        for (int kv_pos = kv_blk; kv_pos < blk_end; kv_pos += kv_step) {
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

            if (skip_compute(kv_pos)) {
                if constexpr (use_causal_mask)
                    causal_left -= kv_step;
                continue;
            }
            b2dK.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+cur_block_id*blk_stride));
            b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            // sometimes KV cache would be filled with random Nan, so need to clean up the unused key data.
            if ((kv_pos + kv_step) > kv_stop) {
                auto valid_rows = kv_stop - kv_pos;
                for (int r = valid_rows; r < kv_step; r++)
                    Kmat.format<half, num_K*REG_M, REG_N>().row(r) = 0.f;
            }
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
        if (first_active) {
            // ugemm_PV0(slm_V, P, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
                // sometimes KV cache would be filled with random Nan, so need to clean up the unused value data.
                if ((kv_pos + kv_step) > kv_stop) {
                    uint valid_rows = kv_stop - kv_pos;
                    uint valid_rows_vnni = (valid_rows+1)/2;
                    for (int r = valid_rows_vnni; r < kv_step / 2; r++)
                        Vmat.row(r) = 0.f;
                    if (valid_rows % 2 == 1)
                        Vmat.row(valid_rows_vnni-1).select<REG_N,2>(1) = 0.f;
                }
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
            first_active = false;
        }
        else {
            //ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
                 // sometimes KV cache would be filled with random Nan, so need to clean up the unused value data.
                if ((kv_pos + kv_step) > kv_stop) {
                    uint valid_rows = kv_stop - kv_pos;
                    uint valid_rows_vnni = (valid_rows+1)/2;
                    for (int r = valid_rows_vnni; r < kv_step / 2; r++)
                        Vmat.row(r) = 0.f;
                    if (valid_rows % 2 == 1)
                        Vmat.row(valid_rows_vnni-1).select<REG_N,2>(1) = 0.f;
                }
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
    }
#else
    // ========================================================================
    // Legacy per-step pipeline for any SPARSE_BLOCK_SIZE (including 1)
    // ======================================================================
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
            if (skip_compute(kv_pos)) {
                if constexpr (use_causal_mask)
                    causal_left -= kv_step;
                continue;
            }
#endif
            b2dK.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+cur_block_id*blk_stride));
            b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(0));
            // sometimes KV cache would be filled with random Nan, so need to clean up the unused key data.
            if ((kv_pos + kv_step) > kv_stop) {
                auto valid_rows = kv_stop - kv_pos;
                for (int r = valid_rows; r < kv_step; r++)
                    Kmat.format<half, num_K*REG_M, REG_N>().row(r) = 0.f;
            }
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
        if (first_active) {
            // ugemm_PV0(slm_V, P, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri = 0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
                // sometimes KV cache would be filled with random Nan, so need to clean up the unused value data.
                if ((kv_pos + kv_step) > kv_stop) {
                    uint valid_rows = kv_stop - kv_pos;
                    uint valid_rows_vnni = (valid_rows+1)/2;
                    for (int r = valid_rows_vnni; r < kv_step / 2; r++)
                        Vmat.row(r) = 0.f;
                    if (valid_rows % 2 == 1)
                        Vmat.row(valid_rows_vnni-1).select<REG_N,2>(1) = 0.f;
                }
                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
            first_active = false;
        }
        else {
            //ugemm_PV1(slm_V, P, max_comp, rO, slm_offset);
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            #pragma unroll
            for(int k = 0, ri=0; k < head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(k));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(k));
                 // sometimes KV cache would be filled with random Nan, so need to clean up the unused value data.
                if ((kv_pos + kv_step) > kv_stop) {
                    uint valid_rows = kv_stop - kv_pos;
                    uint valid_rows_vnni = (valid_rows+1)/2;
                    for (int r = valid_rows_vnni; r < kv_step / 2; r++)
                        Vmat.row(r) = 0.f;
                    if (valid_rows % 2 == 1)
                        Vmat.row(valid_rows_vnni-1).select<REG_N,2>(1) = 0.f;
                }
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
#endif

#ifdef CMPA_DEBUG_ALL_MASKED
    if (first_active) {
        cm_printf("CMPA error: all blocks masked out, q_start=%d\n", q_start);
    }
#endif

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
#endif // CM_HAS_LSC_UNTYPED_2D
