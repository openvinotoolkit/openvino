/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "brgemm_cell_common_fwd.hpp"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "cpu/x64/rnn/brgemm_cell_common_utils.hpp"

using namespace dnnl::impl::utils;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t,
        gemm_acc_t>::brgemm_dst_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const src_t *src_iter,
        const src_t *src_layer, weights_t *w_iter, weights_t *w_layer,
        scratch_t *scratch_gates, gemm_acc_t *amx_scratchpad,
        x64::brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , need_gemm_layer_(rnn_.need_gemm_layer(cell_position))
    , layer_desc_idx_(rnn_.layer_brgemm_desc(cell_position))
    , iter_desc_idx_(rnn_.iter_brgemm_desc(cell_position))
    , Al_(src_layer)
    , Ai_(src_iter)
    , Bl_(w_layer)
    , Bi_(w_iter)
    , C_(scratch_gates)
    , LDAl_(rnn_.src_layer_ld(cell_position))
    , LDAi_(rnn_.src_iter_ld(cell_position))
    , max_nthr_(rnn_.nthr)
    , n_blocking_((rnn_.unfused_post_gemm) ? rnn_.N_blocks * rnn_.n_gates
                                           : rnn_.N_blocks)
    , m_blocking_(rnn_.M_blocks)
    , work_amount_(n_blocking_ * m_blocking_)
    , Bl_n_offset_(rnn_.K1padded * rnn_.n_block)
    , Bi_n_offset_(rnn_.K2padded * rnn_.n_block)
    , Bl_g_offset_(rnn_.N_blocks * Bl_n_offset_)
    , Bi_g_offset_(rnn_.N_blocks * Bi_n_offset_)
    , Al_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block)
    , Ai_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block)
    , Bl_kb_offset_(rnn_.k1_block * rnn_.n_block)
    , Bi_kb_offset_(rnn_.k2_block * rnn_.n_block)
    , Bl_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block * rnn_.n_block)
    , Bi_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block * rnn_.n_block)
    , n_gates_(rnn.unfused_post_gemm ? 1 : rnn.n_gates)
    , brgemm_kernel_iter_main_(need_gemm_layer_
                      ? rnn_brgemm_.kernel_iter_b1_[iter_desc_idx_].get()
                      : rnn_brgemm_.kernel_iter_b0_[iter_desc_idx_].get())
    , brgemm_kernel_iter_n_tail_(need_gemm_layer_
                      ? rnn_brgemm_.kernel_iter_N_tail_b1_[iter_desc_idx_].get()
                      : rnn_brgemm_.kernel_iter_N_tail_b0_[iter_desc_idx_]
                                .get())
    , brgemm_kernel_iter_k_tail_(
              rnn_brgemm_.kernel_iter_K2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_nk_tail_(
              rnn_brgemm_.kernel_iter_NK2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_layer_main_(
              rnn_brgemm_.kernel_layer_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_n_tail_(
              rnn_brgemm_.kernel_layer_N_tail_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_k_tail_(
              rnn_brgemm_.kernel_layer_K1_tail_b1_[layer_desc_idx_].get())
    , brgemm_kernel_layer_nk_tail_(
              rnn_brgemm_.kernel_layer_NK1_tail_b1_[layer_desc_idx_].get())
    , pallete_buff_iter_main_(rnn.k1_block == rnn.k2_block
                      ? rnn_brgemm_.pallete_buff_layer_
                      : rnn_brgemm_.pallete_buff_iter_)
    , pallete_buff_iter_n_tail_(rnn.k1_block == rnn.k2_block
                      ? rnn_brgemm_.pallete_buff_layer_n_tail_
                      : rnn_brgemm_.pallete_buff_iter_n_tail_)
    , pallete_buff_iter_k_tail_(rnn.k1_tail == rnn.k2_tail
                      ? rnn_brgemm_.pallete_buff_k1_tail_
                      : rnn_brgemm_.pallete_buff_k2_tail_)
    , pallete_buff_iter_nk_tail_(rnn.k1_tail == rnn.k2_tail
                      ? rnn_brgemm_.pallete_buff_nk1_tail_
                      : rnn_brgemm_.pallete_buff_nk2_tail_)
    , pallete_buff_layer_main_(rnn_brgemm_.pallete_buff_layer_)
    , pallete_buff_layer_n_tail_(rnn_brgemm_.pallete_buff_layer_n_tail_)
    , pallete_buff_layer_k_tail_(rnn_brgemm_.pallete_buff_k1_tail_)
    , pallete_buff_layer_nk_tail_(rnn_brgemm_.pallete_buff_nk1_tail_)
    , amx_scratchpad_(amx_scratchpad)
    , addr_batch_global_(addr_batch_global)
    , fused_postgemm_(fused_postgemm)
    , is_fused_layer_iter_brgemm_(rnn_.sic == rnn_.slc && LDAi_ == LDAl_) {}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t, gemm_acc_t>::execute()
        const {
    if (is_fused_layer_iter_brgemm_) {
        parallel(max_nthr_, [this](const int ithr, const int nthr) {
            this->kernel_fused_iter_layer(ithr, nthr);
        });
    } else {
        parallel(max_nthr_, [this](const int ithr, const int nthr) {
            this->kernel(ithr, nthr);
        });
    }
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t, gemm_acc_t>::kernel(
        const int ithr, const int nthr) const {

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const bool is_amx = rnn_.is_int8_amx() || rnn_.is_bf16_amx();
    gemm_acc_t *const amx_buffer = is_amx
            ? amx_scratchpad_ + rnn_.m_block * rnn_.n_block * ithr
            : nullptr;
    const int max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
            nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    const char *pallete_buff_iter = nullptr;
    const char *pallete_buff_layer = nullptr;
    const char *pallete_buff_iter_k_tail = nullptr;
    const char *pallete_buff_layer_k_tail = nullptr;

    dim_t nb_i = 0, mb = 0;
    nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);

    amx_tile_configuration_loader_t load_cfg_if_needed;

    while (start < end) {
        const auto m = mb * rnn_.m_block;
        const auto nb = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
        const auto n = nb * rnn_.n_block;
        const auto g_unfused
                = (rnn_.unfused_post_gemm) ? nb_i % rnn_.n_gates : 0;

        const auto *const Al_m = Al_ + m * LDAl_;
        const auto *const Ai_m = Ai_ + m * LDAi_;
        const auto *const Bl_n = Bl_ + nb * Bl_n_offset_;
        const auto *const Bi_n = Bi_ + nb * Bi_n_offset_;
        auto *const C_n = C_ + m * rnn_.LDC + n;

        const brgemm_kernel_t *brgemm_kernel_layer_b0
                = brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_iter = brgemm_kernel_iter_main_;
        const brgemm_kernel_t *brgemm_kernel_layer_k_tail
                = brgemm_kernel_layer_k_tail_;
        const brgemm_kernel_t *brgemm_kernel_iter_k_tail
                = brgemm_kernel_iter_k_tail_;

        if (is_amx) {
            pallete_buff_iter = pallete_buff_iter_main_;
            pallete_buff_layer = pallete_buff_layer_main_;
            pallete_buff_iter_k_tail = pallete_buff_iter_k_tail_;
            pallete_buff_layer_k_tail = pallete_buff_layer_k_tail_;
        }

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;
        if (do_n_tail) {
            brgemm_kernel_layer_b0 = brgemm_kernel_layer_n_tail_;
            brgemm_kernel_iter = brgemm_kernel_iter_n_tail_;
            brgemm_kernel_layer_k_tail = brgemm_kernel_layer_nk_tail_;
            brgemm_kernel_iter_k_tail = brgemm_kernel_iter_nk_tail_;

            if (is_amx) {
                pallete_buff_iter = pallete_buff_iter_n_tail_;
                pallete_buff_layer = pallete_buff_layer_n_tail_;
                pallete_buff_iter_k_tail = pallete_buff_iter_nk_tail_;
                pallete_buff_layer_k_tail = pallete_buff_layer_nk_tail_;
            }
        }

        for (int g = 0; g < n_gates_; g++) {
            const int lg = g + g_unfused;
            const auto *const Bl_g = Bl_n + lg * Bl_g_offset_;
            const auto *const Bi_g = Bi_n + lg * Bi_g_offset_;
            auto *const C_g = C_n + lg * rnn_.N;

            if (need_gemm_layer_) {
                if (is_amx) load_cfg_if_needed(pallete_buff_layer);
                for (int i = 0; i < rnn_.KB1_blocks; i++) {
                    addr_batch[i].ptr.A = Al_m + i * rnn_.k1_block;
                    addr_batch[i].ptr.B = Bl_g + i * Bl_kb_offset_;
                }
                brgemm_kernel_execute(brgemm_kernel_layer_b0, rnn_.KB1_blocks,
                        addr_batch, reinterpret_cast<void *>(C_g), amx_buffer);
            }

            for (int i = 0; i < rnn_.KB2_blocks; i++) {
                addr_batch[i].ptr.A = Ai_m + i * rnn_.k2_block;
                addr_batch[i].ptr.B = Bi_g + i * Bi_kb_offset_;
            }
            if (is_amx) load_cfg_if_needed(pallete_buff_iter);
            brgemm_kernel_execute(brgemm_kernel_iter, rnn_.KB2_blocks,
                    addr_batch, reinterpret_cast<void *>(C_g), amx_buffer);
        }

        if (rnn_.k1_tail && need_gemm_layer_) {
            if (is_amx) load_cfg_if_needed(pallete_buff_layer_k_tail);

            for (int g = 0; g < n_gates_; g++) {
                const int lg = g + g_unfused;
                const auto *const Bl_g = Bl_n + lg * Bl_g_offset_;
                auto *const C_g = C_n + lg * rnn_.N;

                addr_batch[0].ptr.A = Al_m + Al_k_tail_offset_;
                addr_batch[0].ptr.B = Bl_g + Bl_k_tail_offset_;
                brgemm_kernel_execute(brgemm_kernel_layer_k_tail, 1, addr_batch,
                        reinterpret_cast<void *>(C_g), amx_buffer);
            }
        }

        if (rnn_.k2_tail) {
            if (is_amx) load_cfg_if_needed(pallete_buff_iter_k_tail);

            for (int g = 0; g < n_gates_; g++) {
                const int lg = g + g_unfused;
                const auto *const Bi_g = Bi_n + lg * Bi_g_offset_;
                auto *const C_g = C_n + lg * rnn_.N;

                addr_batch[0].ptr.A = Ai_m + Ai_k_tail_offset_;
                addr_batch[0].ptr.B = Bi_g + Bi_k_tail_offset_;
                brgemm_kernel_execute(brgemm_kernel_iter_k_tail, 1, addr_batch,
                        reinterpret_cast<void *>(C_g), amx_buffer);
            }
        }

        if (!rnn_.unfused_post_gemm) {
            const auto block_step = (do_n_tail ? rnn_.n_tail : rnn_.n_block)
                    * sizeof(scratch_t);
            fused_postgemm_(m, n, nb_i, Ai_m, C_n, block_step);
        }

        ++start;
        nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
    }
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t,
        gemm_acc_t>::kernel_fused_iter_layer(const int ithr,
        const int nthr) const {

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const bool is_amx = rnn_.is_int8_amx() || rnn_.is_bf16_amx();
    gemm_acc_t *const amx_buffer = is_amx
            ? amx_scratchpad_ + rnn_.m_block * rnn_.n_block * ithr
            : nullptr;
    const int max_K_Block = 2
            * nstl::max(rnn_.KB1_blocks + 1,
                    nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    const char *pallete_buff = nullptr;
    const char *pallete_buff_k_tail = nullptr;

    dim_t nb_i = 0, mb = 0;
    nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);

    amx_tile_configuration_loader_t load_cfg_if_needed;
    const auto LDA = LDAl_;
    const auto B_n_offset = Bl_n_offset_;
    const auto B_g_offset = Bl_g_offset_;
    const auto B_kb_offset = Bl_kb_offset_;
    const auto KB_blocks
            = (need_gemm_layer_ ? rnn_.KB1_blocks : 0) + rnn_.KB2_blocks;
    const auto KB_blocks_tail = (need_gemm_layer_ ? 1 : 0) + 1;
    const auto A_k_tail_offset = Al_k_tail_offset_;
    const auto B_k_tail_offset = Bl_k_tail_offset_;

    while (start < end) {
        const auto m = mb * rnn_.m_block;
        const auto nb = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
        const auto n = nb * rnn_.n_block;
        const auto g_unfused
                = (rnn_.unfused_post_gemm) ? nb_i % rnn_.n_gates : 0;

        const auto *const Al_m = Al_ + m * LDA;
        const auto *const Ai_m = Ai_ + m * LDA;
        const auto *const Bl_n = Bl_ + nb * B_n_offset;
        const auto *const Bi_n = Bi_ + nb * B_n_offset;
        auto *const C_n = C_ + m * rnn_.LDC + n;

        const brgemm_kernel_t *brgemm_kernel = brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_k_tail
                = brgemm_kernel_layer_k_tail_;

        if (is_amx) {
            pallete_buff = pallete_buff_layer_main_;
            pallete_buff_k_tail = pallete_buff_layer_k_tail_;
        }

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;
        if (do_n_tail) {
            brgemm_kernel = brgemm_kernel_layer_n_tail_;
            brgemm_kernel_k_tail = brgemm_kernel_layer_nk_tail_;

            if (is_amx) {
                pallete_buff = pallete_buff_layer_n_tail_;
                pallete_buff_k_tail = pallete_buff_layer_nk_tail_;
            }
        }

        for (int g = 0; g < n_gates_; g++) {
            const int lg = g + g_unfused;
            const auto *const Bl_g = Bl_n + lg * B_g_offset;
            const auto *const Bi_g = Bi_n + lg * B_g_offset;
            auto *const C_g = C_n + lg * rnn_.N;
            int batch_idx = 0;

            if (need_gemm_layer_) {
                for (; batch_idx < rnn_.KB1_blocks; batch_idx++) {
                    addr_batch[batch_idx].ptr.A
                            = Al_m + batch_idx * rnn_.k1_block;
                    addr_batch[batch_idx].ptr.B
                            = Bl_g + batch_idx * B_kb_offset;
                }
            }

            int iter_idx = 0;
            for (; batch_idx < KB_blocks; batch_idx++) {
                addr_batch[batch_idx].ptr.A = Ai_m + iter_idx * rnn_.k2_block;
                addr_batch[batch_idx].ptr.B = Bi_g + iter_idx * B_kb_offset;
                iter_idx++;
            }

            if (is_amx) load_cfg_if_needed(pallete_buff);
            brgemm_kernel_execute(brgemm_kernel, KB_blocks, addr_batch,
                    reinterpret_cast<void *>(C_g), amx_buffer);
        }

        if (rnn_.k2_tail) {
            for (int g = 0; g < n_gates_; g++) {
                const int lg = g + g_unfused;
                auto *const C_g = C_n + lg * rnn_.N;

                int batch_idx = 0;
                if (need_gemm_layer_) {
                    const auto *const Bl_g = Bl_n + lg * B_g_offset;
                    addr_batch[batch_idx].ptr.A = Al_m + A_k_tail_offset;
                    addr_batch[batch_idx].ptr.B = Bl_g + B_k_tail_offset;
                    batch_idx++;
                }
                const auto *const Bi_g = Bi_n + lg * B_g_offset;
                addr_batch[batch_idx].ptr.A = Ai_m + A_k_tail_offset;
                addr_batch[batch_idx].ptr.B = Bi_g + B_k_tail_offset;

                if (is_amx) load_cfg_if_needed(pallete_buff_k_tail);
                brgemm_kernel_execute(brgemm_kernel_k_tail, KB_blocks_tail,
                        addr_batch, reinterpret_cast<void *>(C_g), amx_buffer);
            }
        }

        if (!rnn_.unfused_post_gemm) {
            const auto block_step = (do_n_tail ? rnn_.n_tail : rnn_.n_block)
                    * sizeof(scratch_t);
            fused_postgemm_(m, n, nb_i, Ai_m, C_n, block_step);
        }

        ++start;
        nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
    }
}

template <typename src_t, typename weights_t, typename gemm_acc_t>
brgemm_dst_proj_t<src_t, weights_t, gemm_acc_t>::brgemm_dst_proj_t(
        const ref_rnn_brgemm_t &rnn_brgemm, const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const src_t *proj_ht,
        const weights_t *w_projection, gemm_acc_t *output,
        gemm_acc_t *amx_scratchpad,
        x64::brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , proj_desc_idx_(
              rnn_.is_f32() ? rnn_.dst_brgemm_desc(cell_position, true) : 0)
    , A_(proj_ht)
    , B_(w_projection)
    , C_(output)
    , LDC_(rnn_.is_f32() ? rnn_.dst_layer_ld(cell_position, true)
                         : rnn_.scratch_gates_ld)
    , max_nthr_(rnn_.nthr)
    , work_amount_proj_(rnn_.Nproj_blocks * rnn_.M_blocks)
    , B_n_offset_(rnn_.Kprojpadded * rnn_.n_block)
    , Bp_kb_offset_(rnn_.kproj_block * rnn_.n_block)
    , amx_scratchpad_(amx_scratchpad)
    , addr_batch_global_(addr_batch_global)
    , brgemm_kernel_main_(rnn_brgemm_.kernel_proj_b0_[proj_desc_idx_].get())
    , brgemm_kernel_n_tail_(
              rnn_brgemm_.kernel_proj_N_tail_b0_[proj_desc_idx_].get())
    , brgemm_kernel_nk_tail_(
              rnn_brgemm_.kernel_proj_NK_tail_b1_[proj_desc_idx_].get())
    , brgemm_kernel_k_tail_(
              rnn_brgemm_.kernel_proj_K_tail_b1_[proj_desc_idx_].get())
    , fused_postgemm_(fused_postgemm) {}

template <typename src_t, typename weights_t, typename gemm_acc_t>
void brgemm_dst_proj_t<src_t, weights_t, gemm_acc_t>::execute() const {
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

template <typename src_t, typename weights_t, typename gemm_acc_t>
void brgemm_dst_proj_t<src_t, weights_t, gemm_acc_t>::kernel(
        const int ithr, const int nthr) const {

    int start = 0, end = 0;
    balance211(work_amount_proj_, nthr, ithr, start, end);
    const bool is_amx = rnn_.is_int8_amx() || rnn_.is_bf16_amx();
    const int max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
            nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    auto *const amx_buffer = is_amx
            ? amx_scratchpad_ + rnn_.m_block * rnn_.n_block * ithr
            : nullptr;
    auto *const addr_batch = is_amx ? addr_batch_global_ + ithr * max_K_Block
                                    : addr_batch_global_ + ithr;
    amx_tile_configuration_loader_t load_cfg_if_needed;

    if (is_amx) load_cfg_if_needed(rnn_brgemm_.pallete_buff_proj_);

    int nb = 0, mb = 0;
    nd_iterator_init(start, nb, rnn_.Nproj_blocks, mb, rnn_.M_blocks);
    while (start < end) {
        const int n = nb * rnn_.n_block;
        const int m = mb * rnn_.m_block;
        const bool do_n_tail = (n + rnn_.n_block) > rnn_.Nproj;
        const int block_step = ((do_n_tail) ? rnn_.nproj_tail : rnn_.n_block)
                * sizeof(src_t);

        const auto *const Ap_m = A_ + m * rnn_.LDAproj;
        const auto *const Bp_n = B_ + nb * B_n_offset_;
        auto *const Cp_n = C_ + m * LDC_ + n;

        const brgemm_kernel_t *const brgemm_kernel_proj_b0
                = do_n_tail ? brgemm_kernel_n_tail_ : brgemm_kernel_main_;

        if (is_amx) {
            if (do_n_tail)
                load_cfg_if_needed(rnn_brgemm_.pallete_buff_nproj_tail_);
            for (int k = 0; k < rnn_.KBproj_blocks; k++) {
                addr_batch[k].ptr.A = Ap_m + k * rnn_.kproj_block;
                addr_batch[k].ptr.B = Bp_n + k * Bp_kb_offset_;
            }
            brgemm_kernel_execute(brgemm_kernel_proj_b0, rnn_.KBproj_blocks,
                    addr_batch, reinterpret_cast<void *>(Cp_n), amx_buffer);

            if (rnn_.kproj_tail) {
                const brgemm_kernel_t *brgemm_kernel_proj_tail;
                const char *tail_cfg_kproj, *tail_recfg;
                if (do_n_tail) {
                    tail_cfg_kproj = rnn_brgemm_.pallete_buff_nkproj_tail_;
                    tail_recfg = rnn_brgemm_.pallete_buff_nproj_tail_;
                    brgemm_kernel_proj_tail = brgemm_kernel_nk_tail_;
                } else {
                    tail_cfg_kproj = rnn_brgemm_.pallete_buff_kproj_tail_;
                    tail_recfg = rnn_brgemm_.pallete_buff_proj_;
                    brgemm_kernel_proj_tail = brgemm_kernel_k_tail_;
                }
                load_cfg_if_needed(tail_cfg_kproj);
                addr_batch[0].ptr.A
                        = Ap_m + rnn_.KBproj_blocks * rnn_.kproj_block;
                addr_batch[0].ptr.B = Bp_n
                        + rnn_.KBproj_blocks * rnn_.kproj_block * rnn_.n_block;
                brgemm_kernel_execute(brgemm_kernel_proj_tail, 1, addr_batch,
                        reinterpret_cast<void *>(Cp_n), amx_buffer);
                load_cfg_if_needed(tail_recfg);
            }
        } else {
            addr_batch[0].ptr.A = Ap_m;
            addr_batch[0].ptr.B = Bp_n;
            brgemm_kernel_execute(brgemm_kernel_proj_b0, 1, addr_batch,
                    reinterpret_cast<void *>(Cp_n), amx_buffer);
        }

        if (!rnn_.unfused_post_gemm) {
            fused_postgemm_(m, n, Cp_n, block_step);
        }

        ++start;
        nd_iterator_step(nb, rnn_.Nproj_blocks, mb, rnn_.M_blocks);
    }
}

template class brgemm_dst_layer_iter_t<uint8_t, int8_t, int32_t, int32_t>;
template class brgemm_dst_layer_iter_t<int8_t, int8_t, int32_t, int32_t>;
template class brgemm_dst_layer_iter_t<float, float, float, float>;
template class brgemm_dst_layer_iter_t<bfloat16_t, bfloat16_t, float, float>;

template class brgemm_dst_proj_t<float, float, float>;
template class brgemm_dst_proj_t<bfloat16_t, bfloat16_t, float>;
template class brgemm_dst_proj_t<int8_t, int8_t, int32_t>;
template class brgemm_dst_proj_t<uint8_t, int8_t, int32_t>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
