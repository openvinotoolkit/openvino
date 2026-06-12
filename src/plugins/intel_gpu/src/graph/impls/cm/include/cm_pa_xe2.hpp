// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CM_HAS_LSC_UNTYPED_2D

#ifndef CMPA_WG_SEQ_LEN
// Keep fixed-WG optimized path optional.
// When undefined, default to 0 so OPTIMIZED_SPARSE_PIPELINE is disabled.
#define CMPA_WG_SEQ_LEN 0
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
    uint slm_St_base,
    int wg_local_id,
    int local_size,
    int q_start,
    int kv_stop,
    int q_tokens_in_tile,
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

    constexpr bool enable_head_size_partition = (head_size == 256);
    constexpr int num_team = enable_head_size_partition ? 8 : 16;
    constexpr int num_worker = 16 / num_team;
    constexpr int process_head_size = head_size / num_worker;

    static_assert(head_size % num_worker == 0, "head_size must be divisible by num_worker");

    int team_id = enable_head_size_partition ? (wg_local_id / num_worker) : wg_local_id;
    int worker_id = enable_head_size_partition ? (wg_local_id % num_worker) : 0;
    int worker_offset = worker_id * process_head_size;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;

    constexpr int num_P_tiles = REG_N / REG_M;
    matrix<half, process_head_size / REG_K, REG_K * REG_N> rQ;
    constexpr int rO_half_rows = process_head_size / 2 / REG_N * num_P_tiles;
    static_assert(process_head_size % (2 * REG_N) == 0, "process_head_size must be divisible by 2*REG_N for rO split");
    matrix<float, rO_half_rows, REG_M * REG_N> rO_lo;
    matrix<float, rO_half_rows, REG_M * REG_N> rO_hi;
    bool first_active = true;

    // clamp per-tile valid query tokens to [0, q_step]
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);
#if KV_CACHE_COMPRESSION == 2
    static_assert(SUB_BLOCK_SIZE % 16 == 0, "SUB_BLOCK_SIZE must be divisible by 16");
    static_assert(CMPA_BLOCK_SZ % SUB_BLOCK_SIZE == 0, "CMPA_BLOCK_SZ must be divisible by SUB_BLOCK_SIZE");
#endif

    if (q_tokens_in_tile < 0) q_tokens_in_tile = 0;
    if (q_tokens_in_tile > q_step) q_tokens_in_tile = q_step;

    // Each worker loads its 1/num_worker chunk of Q
    if (q_tokens_in_tile > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K / 2> b2dQ(
            reinterpret_cast<uint*>(q_base),
            q_tokens_in_tile - 1,
            head_size * sizeof(half) - 1,
            q_pitch - 1,
            0, 0);

        #pragma unroll
        for (int k = 0, ri = 0; k < process_head_size / 2; k += REG_K / 2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(worker_offset / 2 + k));
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
            if (!base) return false;
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

        // Ring slot for this load. Partition path uses a 2-deep ring matching the
        uint slm_offset = enable_head_size_partition
                          ? (slm_buff_id_write & 1) * slm_buff_size
                          : (slm_buff_id_write & 3) * slm_buff_size;

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

            if constexpr (enable_head_size_partition) {
                cm_slm_fence(CM_LOCAL_BARRIER);
                cm_barrier();
            } else {
                if (kv_blk + kv_step < blk_end)
                    load_slm_KV_active(kv_blk + kv_step, blk_end, (kv_blk + kv_step) & cmpa_mask,
                        block_indices[(kv_blk + kv_step) >> cmpa_shift]);

                cm_slm_fence(CM_LOCAL_BARRIER);
                cm_sbarrier(1);
            }

            for (int kv_pos = kv_blk; kv_pos < blk_end; kv_pos += kv_step, slm_buff_id_read++) {
                if constexpr (enable_head_size_partition) {
                    int prefetch_kv_pos = kv_pos + kv_step;
                    load_slm_KV_active(prefetch_kv_pos, blk_end, prefetch_kv_pos & cmpa_mask,
                        block_indices[prefetch_kv_pos >> cmpa_shift]);
                } else {
                    cm_sbarrier(0);

                    // Prefetch 2 steps ahead only if it stays within this block
                    if (kv_pos + 2 * kv_step < blk_end) {
                        int prefetch_kv_pos = kv_pos + 2 * kv_step;
                        load_slm_KV_active(prefetch_kv_pos, blk_end, prefetch_kv_pos & cmpa_mask,
                            block_indices[prefetch_kv_pos >> cmpa_shift]);
                    }

                    if (kv_pos + kv_step < blk_end) {
                        cm_slm_fence(CM_LOCAL_BARRIER);
                        cm_sbarrier(1);
                    }
                }

                {
                    uint slm_offset = enable_head_size_partition ?
                                      (slm_buff_id_read & 1) * slm_buff_size :
                                      (slm_buff_id_read & 3) * slm_buff_size;

                    // Each worker computes partial St using its head_size chunk
                    uint slm_K_worker_offset = slm_offset + worker_offset * kv_step * sizeof(half);
                    matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_K_worker_offset);

                    // Head_size partitioning: synchronize and accumulate partial St
                    if constexpr (enable_head_size_partition) {
                        int slm_offset_bytes = wg_local_id * kv_step * q_step * sizeof(float);
                        cm_slm_block_write(slm_St_base, slm_offset_bytes, St.format<float>());

                        cm_slm_fence(CM_LOCAL_BARRIER);
                        cm_barrier();

                        St = 0.0f;
                        #pragma unroll
                        for (int g = 0; g < num_worker; g++) {
                            int src_wi = team_id * num_worker + g;
                            int src_slm_offset_bytes = src_wi * kv_step * q_step * sizeof(float);
                            matrix<float, kv_step, q_step> partial_st;
                            cm_slm_block_read(slm_St_base, GENX_NONE, src_slm_offset_bytes, partial_st.format<float>());
                            St += partial_st;
                        }
                    }

                    if constexpr (use_causal_mask) {
                        apply_causal_mask_with_offset(St, causal_left);
                        causal_left -= kv_step;
                    }
                    int kv_tokens = kv_stop - kv_pos;
                    for (int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
                    auto max_comp = online_softmax_update(St, cur_max, cur_sum);

                    matrix<half, REG_N, REG_K> P;
                    Transpose2DMatrix(St, P);

                    // Each worker reads its chunk of V from SLM
                    uint slm_V_worker_lo_offset = worker_offset * REG_K * sizeof(half);
                    uint slm_V_worker_hi_offset = (worker_offset + process_head_size / 2) * REG_K * sizeof(half);
                    if (first_active) {
                        ugemm_PV0(slm_V, P, rO_lo, slm_offset + slm_V_worker_lo_offset);
                        ugemm_PV0(slm_V, P, rO_hi, slm_offset + slm_V_worker_hi_offset);
                        first_active = false;
                    } else {
                        ugemm_PV1(slm_V, P, max_comp, rO_lo, slm_offset + slm_V_worker_lo_offset);
                        ugemm_PV1(slm_V, P, max_comp, rO_hi, slm_offset + slm_V_worker_hi_offset);
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
            if (!base) return false;
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

        // Ring slot for this load. Partition path uses a 2-deep ring matching the
        uint slm_offset = enable_head_size_partition
                          ? (slm_buff_id_write & 1) * slm_buff_size
                          : (slm_buff_id_write & 3) * slm_buff_size;
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
    if constexpr (enable_head_size_partition) {
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_barrier();
    } else {
        load_slm_KV(kv_step);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
    }

    for (int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step, slm_buff_id_read++) {
        if constexpr (enable_head_size_partition) {
            load_slm_KV(kv_pos + kv_step);
        } else {
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
            cm_sbarrier(0);

            if (kv_pos + kv_step < kv_stop) {
                cm_slm_fence(CM_LOCAL_BARRIER);
                cm_sbarrier(1);
            }
            load_slm_KV(kv_pos + kv_step * 2);
        }

#if SPARSE_BLOCK_SIZE > 1
        if (skip_compute(kv_pos)) {
            if constexpr (use_causal_mask) {
                causal_left -= kv_step;
            }
            continue;
        }
#endif

        {
            uint slm_offset = enable_head_size_partition ?
                              (slm_buff_id_read & 1) * slm_buff_size :
                              (slm_buff_id_read & 3) * slm_buff_size;

            // Each worker computes partial St using its head_size chunk
            uint slm_K_worker_offset = slm_offset + worker_offset * kv_step * sizeof(half);
            matrix<float, kv_step, q_step> St = ugemm_KQ(slm_K, rQ, slm_K_worker_offset);

            // Head_size partitioning: synchronize and accumulate partial St
            if constexpr (enable_head_size_partition) {
                int slm_offset_bytes = wg_local_id * kv_step * q_step * sizeof(float);
                cm_slm_block_write(slm_St_base, slm_offset_bytes, St.format<float>());

                cm_slm_fence(CM_LOCAL_BARRIER);
                cm_barrier();

                St = 0.0f;
                #pragma unroll
                for (int g = 0; g < num_worker; g++) {
                    int src_wi = team_id * num_worker + g;
                    int src_slm_offset_bytes = src_wi * kv_step * q_step * sizeof(float);
                    matrix<float, kv_step, q_step> partial_st;
                    cm_slm_block_read(slm_St_base, GENX_NONE, src_slm_offset_bytes, partial_st.format<float>());
                    St += partial_st;
                }
            }

            if constexpr (use_causal_mask) {
                apply_causal_mask_with_offset(St, causal_left);
                causal_left -= kv_step;
            }
            int kv_tokens = kv_stop - kv_pos;
            for (int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
            auto max_comp = online_softmax_update(St, cur_max, cur_sum);

            matrix<half, REG_N, REG_K> P;
            Transpose2DMatrix(St, P);

            // Each worker reads its chunk of V from SLM
            uint slm_V_worker_lo_offset = worker_offset * REG_K * sizeof(half);
            uint slm_V_worker_hi_offset = (worker_offset + process_head_size / 2) * REG_K * sizeof(half);
            if (first_active) {
                ugemm_PV0(slm_V, P, rO_lo, slm_offset + slm_V_worker_lo_offset);
                ugemm_PV0(slm_V, P, rO_hi, slm_offset + slm_V_worker_hi_offset);
                first_active = false;
            } else {
                ugemm_PV1(slm_V, P, max_comp, rO_lo, slm_offset + slm_V_worker_lo_offset);
                ugemm_PV1(slm_V, P, max_comp, rO_hi, slm_offset + slm_V_worker_hi_offset);
            }
        }
    }
#endif

    // ============================================================
    // Store O (unchanged)
    // ============================================================
    // U8 path uses workgroup-level barriers in `pa_lsc_u8`, so every lane in
    // the workgroup must participate and cannot early exit.
    if (q_tokens_in_tile == 0) return;

#ifdef CMPA_DEBUG_ALL_MASKED
    if (first_active) {
        cm_printf("CMPA error: all blocks masked out, q_start=%d\n", q_start);
    }
#endif

    matrix<half, num_P_tiles * REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(
        o_base,
        q_tokens_in_tile - 1,
        head_size * sizeof(half) - 1,
        o_pitch - 1,
        0, 0);

    // Store lower half of worker's chunk from rO_lo
    #pragma unroll
    for (int k = 0, ri = 0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {

        #pragma unroll
        for (int p = 0; p < num_P_tiles; p++) {
            auto cO = rO_lo[ri + p].format<float, REG_M, REG_N>();

            #pragma unroll
            for (int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p * REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p * REG_M]);
            }
        }

        int o_offset = worker_offset + k;
        b2dO.set_block_x(o_offset);
        cm_store(b2dO.set_block_y(0),
                 cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M),
                 cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }

    // Store upper half of worker's chunk from rO_hi
    #pragma unroll
    for (int k = process_head_size / 2, ri = 0; k < process_head_size; k += REG_N, ri += num_P_tiles) {

        #pragma unroll
        for (int p = 0; p < num_P_tiles; p++) {
            auto cO = rO_hi[ri + p].format<float, REG_M, REG_N>();

            #pragma unroll
            for (int r = 0; r < cO.n_rows(); r++) {
                cur_O_f16[r + p * REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p * REG_M]);
            }
        }

        int o_offset = worker_offset + k;
        b2dO.set_block_x(o_offset);
        cm_store(b2dO.set_block_y(0),
                 cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
        cm_store(b2dO.set_block_y(REG_M),
                 cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
    }
}

template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
void pa_kernel_lsc_prefetch_u8(
    uint slm_St_base,
    int wg_local_id,
    int q_start,
    int kv_stop,
    int q_tokens_in_tile,
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

    static_assert(head_size == 256, "pa_kernel_lsc_prefetch_u8 supports head_size == 256 only.");

    constexpr uint o_pitch = (num_heads * head_size * sizeof(half));
    constexpr uint q_pitch = is_qkv_fused ? ((num_heads + num_kv_heads * 2) * head_size * sizeof(half)) : o_pitch;
    constexpr uint kv_pitch = head_size * sizeof(uint8_t);

    // K cache layout is various across compression modes.
    //   BY_TOKEN: data + per-token dscale/zp (CMPA_BLOCK_SZ halfs each).
    //   BY_CHANNEL: data + per-(SUB_BLOCK_SIZE × head_size) dscale/zp interleaved.
    // V is always per-token quantized.
#if KV_CACHE_COMPRESSION == 1
    constexpr int k_quan_blk_stride = CMFLA_NUM_KV_HEADS * (CMFLA_HEAD_SIZE + 4) * CMPA_BLOCK_SZ * sizeof(uint8_t);
#else
    constexpr int k_quan_blk_stride = CMFLA_NUM_KV_HEADS * CMFLA_HEAD_SIZE * (CMPA_BLOCK_SZ + CMPA_BLOCK_SZ / SUB_BLOCK_SIZE * 4) * sizeof(uint8_t);
#endif
    constexpr int v_quan_blk_stride = CMFLA_NUM_KV_HEADS * (CMFLA_HEAD_SIZE + 4) * CMPA_BLOCK_SZ * sizeof(uint8_t);

    constexpr bool enable_head_size_partition = (head_size == 256);
    constexpr int num_team = enable_head_size_partition ? 8 : wg_local_size;
    constexpr int num_worker = wg_local_size / num_team;
    constexpr int process_head_size = head_size / num_worker;

    static_assert(wg_local_size == 16, "wg_local_size must be 16");
    static_assert(head_size % num_worker == 0, "head_size must be divisible by num_worker");

    int team_id = enable_head_size_partition ? (wg_local_id / num_worker) : wg_local_id;
    int worker_id = enable_head_size_partition ? (wg_local_id % num_worker) : 0;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;
    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;

    matrix<half, process_head_size / REG_K, REG_K * REG_N> rQ;
    constexpr int rO_half_rows_u8 = process_head_size / 2 / REG_N * num_P_tiles;
    static_assert(process_head_size % (2 * REG_N) == 0, "process_head_size must be divisible by 2*REG_N for rO split");
    matrix<float, rO_half_rows_u8, REG_M * REG_N> rO_lo;
    matrix<float, rO_half_rows_u8, REG_M * REG_N> rO_hi;
    bool first_active = true;

#if SPARSE_BLOCK_SIZE > 1
    constexpr int sb_shift = (SPARSE_BLOCK_SIZE == 128) ? 7 : (SPARSE_BLOCK_SIZE == 256) ? 8 : -1;
    auto skip_by = [&](const bool* base, int kv_pos) -> bool {
        if constexpr (sb_shift < 0) {
            return false;
        } else {
            if (!base) return false;
            return !base[(uint)kv_pos >> sb_shift];
        }
    };
    auto skip_compute = [&](int kv_pos) {
        return skip_by((const bool*)sparse_mask_base, kv_pos);
    };
#endif

    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_in_tile < 0) q_tokens_in_tile = 0;
    if (q_tokens_in_tile > q_step) q_tokens_in_tile = q_step;

    if constexpr (!enable_head_size_partition) {
        if (q_tokens_in_tile == 0) return;
    }

    int worker_offset = worker_id * process_head_size;
    if (q_tokens_in_tile > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_in_tile - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for (int k = 0, ri = 0; k < process_head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(worker_offset / 2 + k));
            rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
        }
    }

    lsc::block_2d_desc<uint8_t, 1, kv_step, REG_K> b2dK(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<uint8_t, 1, REG_K, REG_N> b2dV(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);

    static_assert(wg_local_size == 16);
    lsc::block_2d_desc<uint8_t, 1, kv_step/wg_local_size, REG_K> prefetch_K(k_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
    lsc::block_2d_desc<uint8_t, 1, REG_K/wg_local_size, REG_N> prefetch_V(v_cache_base, CMPA_BLOCK_SZ - 1, head_size*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);

    int causal_left = q_start + past_lens;

    auto slm_St = slm_St_base;

    for (int kv_pos = 0; kv_pos < kv_stop; kv_pos += kv_step) {
        auto cur_block_id = block_indices[kv_pos / CMPA_BLOCK_SZ];
        uint32_t prefetch_kv_pos = (kv_pos + kv_step) >= kv_stop ? kv_pos : (kv_pos + kv_step);
        auto prefetch_block_id = block_indices[prefetch_kv_pos / CMPA_BLOCK_SZ];

        matrix<float, kv_step, q_step> St;
        {
            constexpr int num_K = kv_step / REG_M;
            auto St2 = St.format<float, num_K, REG_M*REG_N>();

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

            vector<half, kv_step> k_dscale;
            vector<half, kv_step> k_zp;
#if KV_CACHE_COMPRESSION == 1
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_dscale_offset), k_dscale);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_zp_offset), k_zp);
#endif

            matrix<uint8_t, kv_step, REG_K> quanKmat;
            matrix<half, num_K, REG_M * REG_K> Kmat;

            prefetch_K.set_base_ptr((reinterpret_cast<uint8_t*>(k_cache_base) + prefetch_block_id * k_quan_blk_stride));
            prefetch_K.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(worker_offset));

#if SPARSE_BLOCK_SIZE > 1
            if (skip_compute(kv_pos)) {
                if constexpr (use_causal_mask)
                    causal_left -= kv_step;
                continue;
            }
#endif
            b2dK.set_base_ptr((reinterpret_cast<uint8_t*>(k_cache_base) + cur_block_id * k_quan_blk_stride));
            b2dK.set_block_y(kv_pos % CMPA_BLOCK_SZ);

            // First K tile: column offset = worker_offset.
#if KV_CACHE_COMPRESSION == 2
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_dscale_offset + worker_offset * sizeof(half)), k_dscale);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_zp_offset    + worker_offset * sizeof(half)), k_zp);
#endif
            cm_load<lsc::Normal>(quanKmat.format<uint8_t>(), b2dK.set_block_x(worker_offset));

            {
                auto Kmat_flat = Kmat.format<half, kv_step, REG_K>();
                #pragma unroll
                for (int r = 0; r < kv_step; r++) {
#if KV_CACHE_COMPRESSION == 1
                    Kmat_flat[r] = quanKmat[r] - k_zp[r];
                    Kmat_flat[r] = cm_mul<half>(Kmat_flat[r], k_dscale[r]);
#else
                    Kmat_flat[r] = quanKmat[r] - k_zp;
                    Kmat_flat[r] = cm_mul<half>(Kmat_flat[r], k_dscale);
#endif
                }
            }

            if ((kv_pos + kv_step) > kv_stop) {
                auto valid_rows = kv_stop - kv_pos;
                for (int r = valid_rows; r < kv_step; r++)
                    Kmat.format<half, num_K*REG_M, REG_N>().row(r) = 0.f;
            }

            #pragma unroll
            for (int k = 0; k < num_K; k++)
                St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0,
                                rQ[0].format<int32_t>(),
                                Kmat[k].format<int32_t>());

            #pragma unroll
            for (int ri = 1; ri < process_head_size/REG_K; ri++) {
                int k_offset = worker_offset + ri*REG_K;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(k_offset));
#if KV_CACHE_COMPRESSION == 2
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_dscale_offset + k_offset * sizeof(half)), k_dscale);
                cm_svm_block_read(reinterpret_cast<svmptr_t>(k_cache_base + k_zp_offset    + k_offset * sizeof(half)), k_zp);
#endif
                cm_load<lsc::Normal>(quanKmat.format<uint8_t>(), b2dK.set_block_x(k_offset));
                {
                    auto Kmat_flat = Kmat.format<half, kv_step, REG_K>();
                    #pragma unroll
                    for (int r = 0; r < kv_step; r++) {
#if KV_CACHE_COMPRESSION == 1
                        Kmat_flat[r] = quanKmat[r] - k_zp[r];
                        Kmat_flat[r] = cm_mul<half>(Kmat_flat[r], k_dscale[r]);
#else
                        Kmat_flat[r] = quanKmat[r] - k_zp;
                        Kmat_flat[r] = cm_mul<half>(Kmat_flat[r], k_dscale);
#endif
                    }
                }
                if ((kv_pos + kv_step) > kv_stop) {
                    auto valid_rows = kv_stop - kv_pos;
                    for (int r = valid_rows; r < kv_step; r++)
                        Kmat.format<half, num_K*REG_M, REG_N>().row(r) = 0.f;
                }
                #pragma unroll
                for (int k = 0; k < num_K; k++) {
                    St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St2.row(k),
                        rQ[ri].format<int32_t>(),
                        Kmat[k].format<int32_t>());
                }
            }

            // Head_size partitioning: synchronize and accumulate partial St across workers
            if constexpr (enable_head_size_partition) {
                int slm_offset_bytes = wg_local_id * kv_step * q_step * sizeof(float);
                cm_slm_block_write(slm_St, slm_offset_bytes, St.format<float>());

                cm_slm_fence(CM_LOCAL_BARRIER);
                cm_barrier();

                St = 0.0f;
                #pragma unroll
                for (int g = 0; g < num_worker; g++) {
                    int src_wi = team_id * num_worker + g;
                    int src_slm_offset_bytes = src_wi * kv_step * q_step * sizeof(float);
                    matrix<float, kv_step, q_step> partial_st;
                    cm_slm_block_read(slm_St, GENX_NONE, src_slm_offset_bytes, partial_st.format<float>());
                    St += partial_st;
                }
            }
        }

        if constexpr (use_causal_mask) {
            apply_causal_mask_with_offset(St, causal_left);
            causal_left -= kv_step;
        }
        int kv_tokens = kv_stop - kv_pos;
        for (int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
        auto max_comp = online_softmax_update(St, cur_max, cur_sum);

        matrix<half, REG_N, REG_K> P;
        Transpose2DMatrix(St, P);

        int kv_pos_in_block_v = kv_pos - (kv_pos / CMPA_BLOCK_SZ) * CMPA_BLOCK_SZ;
        uint32_t v_dscale_offset =
            cur_block_id * v_quan_blk_stride +
            CMPA_BLOCK_SZ * head_size * sizeof(uint8_t) +
            kv_pos_in_block_v * sizeof(half);
        uint32_t v_zp_offset = v_dscale_offset + CMPA_BLOCK_SZ * sizeof(half);
        vector<half, kv_step> v_dscale;
        vector<half, kv_step> v_zp;
        cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + v_dscale_offset), v_dscale);
        cm_svm_block_read(reinterpret_cast<svmptr_t>(v_cache_base + v_zp_offset), v_zp);

        prefetch_V.set_base_ptr((reinterpret_cast<uint8_t*>(v_cache_base) + prefetch_block_id * v_quan_blk_stride));
        prefetch_V.set_block_y((prefetch_kv_pos + wg_local_id) % CMPA_BLOCK_SZ);

        b2dV.set_base_ptr((reinterpret_cast<uint8_t*>(v_cache_base) + cur_block_id * v_quan_blk_stride));
        b2dV.set_block_y(kv_pos % CMPA_BLOCK_SZ);

        if (first_active) {
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            // PV0 lower half
            #pragma unroll
            for (int k = 0, ri = 0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                {
                    matrix<uint8_t, REG_K, REG_N> quanVmat;
                    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                    cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(v_offset));
                    matrix<half, REG_K, REG_N> Vraw;
                    #pragma unroll
                    for (int r = 0; r < REG_K; r++) {
                        Vraw[r] = quanVmat[r] - v_zp[r];
                        Vraw[r] = cm_mul<half>(Vraw[r], v_dscale[r]);
                    }
                    if ((kv_pos + kv_step) > kv_stop) {
                        uint valid_rows = kv_stop - kv_pos;
                        for (int r = valid_rows; r < REG_K; r++)
                            Vraw[r] = 0.f;
                    }
                    prepackAsVNNIWidth2(Vraw, Vmat);
                }
                #pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    rO_lo[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
            // PV0 upper half
            #pragma unroll
            for (int k = process_head_size / 2, ri = 0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                {
                    matrix<uint8_t, REG_K, REG_N> quanVmat;
                    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                    cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(v_offset));
                    matrix<half, REG_K, REG_N> Vraw;
                    #pragma unroll
                    for (int r = 0; r < REG_K; r++) {
                        Vraw[r] = quanVmat[r] - v_zp[r];
                        Vraw[r] = cm_mul<half>(Vraw[r], v_dscale[r]);
                    }
                    if ((kv_pos + kv_step) > kv_stop) {
                        uint valid_rows = kv_stop - kv_pos;
                        for (int r = valid_rows; r < REG_K; r++)
                            Vraw[r] = 0.f;
                    }
                    prepackAsVNNIWidth2(Vraw, Vmat);
                }
                #pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    rO_hi[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
            first_active = false;
        } else {
            auto P2 = P.format<half, num_P_tiles, REG_M * REG_K>();
            // PV1 lower half
            #pragma unroll
            for (int k = 0, ri = 0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                {
                    matrix<uint8_t, REG_K, REG_N> quanVmat;
                    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                    cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(v_offset));
                    matrix<half, REG_K, REG_N> Vraw;
                    #pragma unroll
                    for (int r = 0; r < REG_K; r++) {
                        Vraw[r] = quanVmat[r] - v_zp[r];
                        Vraw[r] = cm_mul<half>(Vraw[r], v_dscale[r]);
                    }
                    if ((kv_pos + kv_step) > kv_stop) {
                        uint valid_rows = kv_stop - kv_pos;
                        for (int r = valid_rows; r < REG_K; r++)
                            Vraw[r] = 0.f;
                    }
                    prepackAsVNNIWidth2(Vraw, Vmat);
                }
                #pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    auto cO = rO_lo[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for (int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                }
                #pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    rO_lo[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_lo[ri + p].format<float>(),
                                Vmat.format<int32_t>(),
                                P2.row(p).format<int32_t>());
                }
            }
            // PV1 upper half
            #pragma unroll
            for (int k = process_head_size / 2, ri = 0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                {
                    matrix<uint8_t, REG_K, REG_N> quanVmat;
                    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                    cm_load<lsc::Normal>(quanVmat.format<uint8_t>(), b2dV.set_block_x(v_offset));
                    matrix<half, REG_K, REG_N> Vraw;
                    #pragma unroll
                    for (int r = 0; r < REG_K; r++) {
                        Vraw[r] = quanVmat[r] - v_zp[r];
                        Vraw[r] = cm_mul<half>(Vraw[r], v_dscale[r]);
                    }
                    if ((kv_pos + kv_step) > kv_stop) {
                        uint valid_rows = kv_stop - kv_pos;
                        for (int r = valid_rows; r < REG_K; r++)
                            Vraw[r] = 0.f;
                    }
                    prepackAsVNNIWidth2(Vraw, Vmat);
                }
                #pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    auto cO = rO_hi[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for (int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                }
                #pragma unroll
                for (int p = 0; p < num_P_tiles; p++) {
                    rO_hi[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_hi[ri + p].format<float>(),
                                Vmat.format<int32_t>(),
                                P2.row(p).format<int32_t>());
                }
            }
        }
    }

#ifdef CMPA_DEBUG_ALL_MASKED
    if (first_active) {
        cm_printf("CMPA error: all blocks masked out, q_start=%d\n", q_start);
    }
#endif

    matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    if (q_tokens_in_tile > 0) {
        lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_in_tile - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);

        // Store lower half from rO_lo
        #pragma unroll
        for (int k = 0, ri = 0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
            #pragma unroll
            for (int p = 0; p < num_P_tiles; p++) {
                auto cO = rO_lo[ri + p].format<float, REG_M, REG_N>();
                #pragma unroll
                for (int r = 0; r < cO.n_rows(); r++) {
                    cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
                }
            }
            int o_offset = worker_offset + k;
            b2dO.set_block_x(o_offset);
            cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
            cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
        }

        // Store upper half from rO_hi
        #pragma unroll
        for (int k = process_head_size / 2, ri = 0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
            #pragma unroll
            for (int p = 0; p < num_P_tiles; p++) {
                auto cO = rO_hi[ri + p].format<float, REG_M, REG_N>();
                #pragma unroll
                for (int r = 0; r < cO.n_rows(); r++) {
                    cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
                }
            }
            int o_offset = worker_offset + k;
            b2dO.set_block_x(o_offset);
            cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
            cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
        }
    }
}

#else

template<bool use_causal_mask, int num_heads, int num_kv_heads, int head_size, int is_qkv_fused, int wg_local_size>
void pa_kernel_lsc_prefetch_f16(
    uint slm_St_base,
    int wg_local_id,
    int q_start,
    int kv_stop, //
    int q_tokens_in_tile, // number of valid query tokens in this q_step tile
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

    constexpr bool enable_head_size_partition = (head_size == 256);
    constexpr int num_team = enable_head_size_partition ? 8 : wg_local_size;
    constexpr int num_worker = wg_local_size / num_team;
    constexpr int process_head_size = head_size / num_worker;

    static_assert(wg_local_size == 16, "wg_local_size must be 16");
    static_assert(head_size % num_worker == 0, "head_size must be divisible by num_worker");

    int team_id = enable_head_size_partition ? (wg_local_id / num_worker) : wg_local_id;
    int worker_id = enable_head_size_partition ? (wg_local_id % num_worker) : 0;

    vector<float, q_step> cur_max;
    vector<float, q_step> cur_sum;

    cur_max = -3e38f;
    cur_sum = 0;
    constexpr int num_P_tiles = REG_N / REG_M;

    // Each worker only allocates 1/4 of head_size
    matrix<half, process_head_size/REG_K, REG_K*REG_N> rQ;
    constexpr int rO_half_rows_f16 = process_head_size / 2 / REG_N * num_P_tiles;
    static_assert(process_head_size % (2 * REG_N) == 0, "process_head_size must be divisible by 2*REG_N for rO split");
    matrix<float, rO_half_rows_f16, REG_M * REG_N> rO_lo;
    matrix<float, rO_half_rows_f16, REG_M * REG_N> rO_hi;
    bool first_active = true;

#if SPARSE_BLOCK_SIZE > 1
    constexpr int sb_shift = (SPARSE_BLOCK_SIZE == 128) ? 7 : (SPARSE_BLOCK_SIZE == 256) ? 8 : -1;
    auto skip_by = [&](const bool* base, int kv_pos) -> bool {
        if constexpr (sb_shift < 0) {
            return false;
        } else {
            if (!base) return false;
            return !base[(uint)kv_pos >> sb_shift];
        }
    };

    auto skip_compute = [&](int kv_pos) {
        return skip_by((const bool*)sparse_mask_base, kv_pos);
    };
#endif

    // clamp per-tile valid query tokens to [0, q_step]
    static_assert(q_step == REG_N);
    static_assert(kv_step == REG_K);

    if (q_tokens_in_tile < 0) q_tokens_in_tile = 0;
    if (q_tokens_in_tile > q_step) q_tokens_in_tile = q_step;

    // Threads with zero valid query tokens can early exit if there is no barrier.
    if constexpr (!enable_head_size_partition) {
        if (q_tokens_in_tile == 0) return;
    }

    // Each worker loads its 1/num_worker chunk of Q
    int worker_offset = worker_id * process_head_size;
    if (q_tokens_in_tile > 0) {
        lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dQ(reinterpret_cast<uint*>(q_base), q_tokens_in_tile - 1, head_size*sizeof(half) - 1, q_pitch - 1, 0, 0);
        #pragma unroll
        for(int k = 0, ri = 0; k < process_head_size/2; k += REG_K/2, ri++) {
            cm_load<lsc::Transpose>(rQ[ri].format<uint>(), b2dQ.set_block_x(worker_offset / 2 + k));
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
    // Use SLM passed from kernel for accumulating partial attention scores across workers
    constexpr int slm_size_per_wi = kv_step * q_step;  // St matrix size
    auto slm_St = slm_St_base;

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
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(worker_offset));

            if (skip_compute(kv_pos)) {
                if constexpr (use_causal_mask)
                    causal_left -= kv_step;
                continue;
            }
            b2dK.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+cur_block_id*blk_stride));
            b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(worker_offset));
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
            for(int ri = 1; ri < process_head_size/REG_K; ri++) {
                int k_offset = worker_offset + ri*REG_K;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(k_offset));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(k_offset));
                #pragma unroll
                for(int k = 0; k < num_K; k++) {
                    St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St2.row(k),
                        rQ[ri].format<int32_t>(),
                        Kmat[k].format<int32_t>());
                }
            }

        // Head_size partitioning: synchronize and accumulate partial St
        // Work-items with same team_id accumulate across head_size chunks
        if constexpr (enable_head_size_partition) {
            // Store partial St to SLM for this work-item
            int slm_offset_bytes = wg_local_id * kv_step * q_step * sizeof(float);
            cm_slm_block_write(slm_St, slm_offset_bytes, St.format<float>());

            // Barrier: ensure all work-items have written their partial St
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_barrier();

            // Accumulate partial St from all 4 head_size chunks for this query slice
            // Work-items [team_id*4, team_id*4+1, team_id*4+2, team_id*4+3] cooperate
            St = 0.0f;
            #pragma unroll
            for(int g = 0; g < num_worker; g++) {
                int src_wi = team_id * num_worker + g;  // Same query slice (team_id), different head_size chunk (g)
                int src_slm_offset_bytes = src_wi * kv_step * q_step * sizeof(float);
                matrix<float, kv_step, q_step> partial_st;
                cm_slm_block_read(slm_St, GENX_NONE, src_slm_offset_bytes, partial_st.format<float>());
                St += partial_st;
            }
        }

        }
        if constexpr (use_causal_mask) {
            apply_causal_mask_with_offset(St, causal_left);
            causal_left -= kv_step;
        }
        int kv_tokens = kv_stop - kv_pos;
        // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
        for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
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
            // PV0 lower half - Each worker loads its 1/4 chunk of V
            #pragma unroll
            for(int k = 0, ri = 0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    rO_lo[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
            // PV0 upper half - Second half of this worker's chunk
            #pragma unroll
            for(int k = process_head_size / 2, ri = 0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    rO_hi[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
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
            // PV1 lower half - Each worker loads its 1/4 chunk of V
            #pragma unroll
            for(int k = 0, ri=0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    auto cO = rO_lo[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                }

                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO_lo[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_lo[ri + p].format<float>(),
                                Vmat.format<int32_t>(),
                                P2.row(p).format<int32_t>());
                }
            }
            // PV1 upper half - Second half of this worker's chunk
            #pragma unroll
            for(int k = process_head_size / 2, ri=0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    auto cO = rO_hi[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                }

                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO_hi[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_hi[ri + p].format<float>(),
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
    // Use SLM passed from kernel for accumulating partial attention scores across workers
    constexpr int slm_size_per_wi = kv_step * q_step;  // St matrix size
    auto slm_St = slm_St_base;

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
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(worker_offset));

#if SPARSE_BLOCK_SIZE > 1
            if (skip_compute(kv_pos)) {
                if constexpr (use_causal_mask)
                    causal_left -= kv_step;
                continue;
            }
#endif
            b2dK.set_base_ptr((reinterpret_cast<half*>(k_cache_base)+cur_block_id*blk_stride));
            b2dK.set_block_y(kv_pos%CMPA_BLOCK_SZ);

            // Each work-item loads its 1/4 chunk of K and computes partial St
            cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(worker_offset));
            // sometimes KV cache would be filled with random Nan, so need to clean up the unused key data.
            if ((kv_pos + kv_step) > kv_stop) {
                auto valid_rows = kv_stop - kv_pos;
                for (int r = valid_rows; r < kv_step; r++)
                    Kmat.format<half, num_K*REG_M, REG_N>().row(r) = 0.f;
            }

            // Compute partial St with this worker's Q chunk
            #pragma unroll
            for(int k = 0; k < num_K; k++)
                St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                0,
                                rQ[0].format<int32_t>(),
                                Kmat[k].format<int32_t>());

            #pragma unroll
            for(int ri = 1; ri < process_head_size/REG_K; ri++) {
                int k_offset = worker_offset + ri*REG_K;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_K.set_block_x(k_offset));
                cm_load<lsc::Normal>(Kmat.format<half>(), b2dK.set_block_x(k_offset));
                #pragma unroll
                for(int k = 0; k < num_K; k++) {
                    St2.row(k) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                        St2.row(k),
                        rQ[ri].format<int32_t>(),
                        Kmat[k].format<int32_t>());
                }
            }

        // Head_size partitioning: synchronize and accumulate partial St
        // Work-items with same team_id (processing same query slice) accumulate across head_size chunks
        if constexpr (enable_head_size_partition) {
            // Store partial St to SLM for this work-item
            int slm_offset_bytes = wg_local_id * kv_step * q_step * sizeof(float);
            cm_slm_block_write(slm_St, slm_offset_bytes, St.format<float>());

            // Barrier: ensure all work-items have written their partial St
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_barrier();

            // Accumulate partial St from all 4 head_size chunks for this query slice
            // Work-items [team_id*4, team_id*4+1, team_id*4+2, team_id*4+3] cooperate
            St = 0.0f;
            #pragma unroll
            for(int g = 0; g < num_worker; g++) {
                int src_wi = team_id * num_worker + g;  // Same query slice (team_id), different head_size chunk (g)
                int src_slm_offset_bytes = src_wi * kv_step * q_step * sizeof(float);
                matrix<float, kv_step, q_step> partial_st;
                cm_slm_block_read(slm_St, GENX_NONE, src_slm_offset_bytes, partial_st.format<float>());
                St += partial_st;
            }
        }

        }
        if constexpr (use_causal_mask) {
            apply_causal_mask_with_offset(St, causal_left);
            causal_left -= kv_step;
        }
        int kv_tokens = kv_stop - kv_pos;
        // LSC ensures no overflow-access, but mask off k-tails attn-score is still required
        for(int p = kv_tokens; p < kv_step; p++) St[p] = -3.4e38f;
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
            // PV0 lower half - Each worker loads its 1/4 chunk of V
            #pragma unroll
            for(int k = 0, ri = 0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    rO_lo[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
                                    0,
                                    Vmat.format<int32_t>(),
                                    P2.row(p).format<int32_t>());
                }
            }
            // PV0 upper half - Second half of this worker's chunk
            #pragma unroll
            for(int k = process_head_size / 2, ri = 0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;
                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    rO_hi[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount, float>(
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
            // PV1 lower half - Each worker loads its 1/4 chunk of V
            #pragma unroll
            for(int k = 0, ri=0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    auto cO = rO_lo[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                }

                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO_lo[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_lo[ri + p].format<float>(),
                                Vmat.format<int32_t>(),
                                P2.row(p).format<int32_t>());
                }
            }
            // PV1 upper half - Second half of this worker's chunk
            #pragma unroll
            for(int k = process_head_size / 2, ri=0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
                matrix<half, REG_K/2, REG_N*2> Vmat;

                int v_offset = worker_offset + k;
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(prefetch_V.set_block_x(v_offset));
                cm_load<lsc::VNNI>(Vmat.format<half>(), b2dV.set_block_x(v_offset));
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
                    auto cO = rO_hi[ri + p].format<float, REG_M, REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_M; r++)
                        cO.row(r) = cm_mul<float>(cO.row(r), max_comp[r + p*REG_M]);
                }

                #pragma unroll
                for(int p = 0; p < num_P_tiles; p++) {
                    rO_hi[ri + p] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                                rO_hi[ri + p].format<float>(),
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

    //# save cur_O/cur_sum.transpose(0, 1)
    matrix<half, num_P_tiles*REG_M, REG_N> cur_O_f16;
    cur_sum = cm_inv(cur_sum);

    // Inactive threads (q_tokens_in_tile==0) participated in barriers but must not write output.
    if (q_tokens_in_tile > 0) {
        lsc::block_2d_desc<half, 1, REG_M, REG_N> b2dO(o_base, q_tokens_in_tile - 1, head_size*sizeof(half) - 1, o_pitch - 1, 0, 0);

        // Each worker stores its 1/4 chunk of output
        // Store lower half of worker's chunk from rO_lo
        #pragma unroll
        for(int k = 0, ri=0; k < process_head_size / 2; k += REG_N, ri += num_P_tiles) {
            #pragma unroll
            for(int p = 0; p < num_P_tiles; p++) {
                auto cO = rO_lo[ri + p].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < cO.n_rows(); r++) {
                    cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
                }
            }
            int o_offset = worker_offset + k;
            b2dO.set_block_x(o_offset);
            cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
            cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
        }

        // Store upper half of worker's chunk from rO_hi
        #pragma unroll
        for(int k = process_head_size / 2, ri=0; k < process_head_size; k += REG_N, ri += num_P_tiles) {
            #pragma unroll
            for(int p = 0; p < num_P_tiles; p++) {
                auto cO = rO_hi[ri + p].format<float, REG_M, REG_N>();
                #pragma unroll
                for(int r = 0; r < cO.n_rows(); r++) {
                    cur_O_f16[r + p*REG_M] = cm_mul<float>(cO.row(r), cur_sum[r + p*REG_M]);
                }
            }
            int o_offset = worker_offset + k;
            b2dO.set_block_x(o_offset);
            cm_store(b2dO.set_block_y(0), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(0));
            cm_store(b2dO.set_block_y(REG_M), cur_O_f16.format<half, num_P_tiles, REG_M * REG_N>().row(1));
        }
    }
}

#endif
#endif // CM_HAS_LSC_UNTYPED_2D
