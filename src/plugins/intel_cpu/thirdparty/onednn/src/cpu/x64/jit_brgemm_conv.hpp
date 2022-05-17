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

#ifndef CPU_X64_JIT_BRGEMM_CONV_HPP
#define CPU_X64_JIT_BRGEMM_CONV_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct brgemm_convolution_fwd_t : public primitive_t {

    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , with_sum(false) {}

        ~pd_t() = default;

        // ------- DECLARE_COMMON_PD_t -----
        pd_t *clone() const override {
            auto new_pd = utils::make_unique<pd_t>(*this);
            if (!new_pd->is_initialized()) return nullptr;
            new_pd->brgs_.resize(brgs_sz_);
            for (int i = 0; i < brgs_sz_; i++) {
                new_pd->brgs_[i] = brgs_[i];
                new_pd->bd_masks[i] = bd_masks[i];
            }
            return new_pd.release();
        }

        status_t create_primitive(
                std::pair<std::shared_ptr<primitive_t>, bool> &primitive,
                engine_t *engine) const override {
            return primitive_t::create_primitive_common<
                    brgemm_convolution_fwd_t, pd_t>(
                    primitive, this, engine, false);
        }

        const char *name() const override {
            return JIT_IMPL_NAME_HELPER("brgconv:", isa, "");
        }
        // ---------------------------------

        status_t init(engine_t *engine);

        int brgs_sz_;
        std::vector<brgemm_t> brgs_;
        std::vector<std::shared_ptr<std::vector<char>>> bd_masks;
        bool with_sum;
        jit_brgemm_conv_conf_t jcp_;
        int bs_b, bs_e, bs_s, bs_c; // batch size info for unrolled kernels
        int get_brg_idx(int bs, int m, bool do_initialization, bool is_N_tail,
                bool is_K_tail) const {
            auto adj_bs = jcp_.use_uker ? (bs / bs_s) - 1 : 0;
            return (((m * bs_c + adj_bs) * 2
                            + static_cast<int>(do_initialization))
                                   * 2
                           + static_cast<int>(is_N_tail))
                    * 2
                    + static_cast<int>(is_K_tail);
        }
    };

    brgemm_convolution_fwd_t(const pd_t *apd);

    ~brgemm_convolution_fwd_t() = default;

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init(engine_t *engine) override;

private:
    struct S_t {
        char a[AMX_PALETTE_SIZE];
    };

    //  brgemm convolution execution context
    struct brgemm_exec_ctx_t {
        brgemm_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd)
            : src(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
            , weights(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
            , bias(CTX_IN_MEM(const char *, DNNL_ARG_BIAS))
            , dst(CTX_OUT_MEM(char *, DNNL_ARG_DST))
            , post_ops_binary_rhs_arg_vec(binary_injector::prepare_binary_args(
                      pd->attr()->post_ops_, ctx)) {}
        const char *const __restrict src;
        const char *const __restrict weights;
        const char *const __restrict bias;
        char *const __restrict dst;
        const std::vector<const void *> post_ops_binary_rhs_arg_vec;
    };

    struct brgemm_thread_ctx_t {
        brgemm_thread_ctx_t(brgemm_exec_ctx_t &brgemm_ctx_, int ithr_,
                brgemm_batch_element_t *__restrict brg_batch_, char *c_buffer_,
                char *wsp_tile_)
            : brgemm_ctx(brgemm_ctx_)
            , ithr(ithr_)
            , brg_batch(brg_batch_)
            , c_buffer(c_buffer_)
            , wsp_tile(wsp_tile_) {}

        brgemm_exec_ctx_t &brgemm_ctx;
        int ithr;
        brgemm_batch_element_t *__restrict brg_batch;
        char *c_buffer;
        char *wsp_tile;
        S_t cur_palette;
        int g, n, ocb;
        int od, odb, oh, ohb, owb;
        int icc;
    };

    static int get_ker_po_idx(int m, bool do_postwork, bool is_N_tail) {
        return (m * 2 + static_cast<int>(do_postwork)) * 2
                + static_cast<int>(is_N_tail);
    }

    static int get_inp_size(
            int max_src_size, int dst_size, int k, int stride, int dilate) {
        const auto res = nstl::min(max_src_size,
                calculate_end_padding(0, dst_size, 0, stride,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    void get_kw_range(
            int ow, int &kw_s, int &kw_full_s, int &kw_full_e, int &kw_e) const;
    void get_ow_range(int ow, int kw, int &ow_s, int &ow_e) const;

    void ker_base(brgemm_thread_ctx_t &btc) const;
    void ker_trans(brgemm_thread_ctx_t &btc, char *inp_buffer) const;
    void ker_vpad(brgemm_thread_ctx_t &btc) const;

    void perform_outwork(char *dst_base, char *c_buffer, const char *bias_w,
            int od, int oh, int ow, int g_oc, bool is_oc_tail, int ker_ow_s,
            int ker_ow_f, int kd_l, int kh_l,
            const void *post_ops_binary_rhs_arg_vec, bool maybe_do_init,
            bool do_postwork) const;

    void call_brgemm_kernel(brgemm_thread_ctx_t &btc, int brg_idx,
            int batch_size, char *ptr_C, char *ptr_D, const char *bias_w,
            int g_oc, bool do_postops, const void *binary_post_ops_rhs) const;

    void maybe_conv_inp(int ithr, const char *__restrict src,
            char *__restrict inp_buffer, uint8_t *__restrict inp_buffer_mask,
            int g, int n, int icc, int odb, int ohb, int owb, int last_g,
            int last_n, int last_icc, int last_odb, int last_ohb,
            int last_owb) const;

    status_t add_po_kernel(brgemm_t &bcfg, int ker_idx, bool is_init);
    void add_po_kernels(
            int i_N, int init_bcast_dim, int po_bcast_dim, bool need_postwork);
    status_t add_brg_kernel(int bs, int M, int i_N, int i_K, int i_init);

    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    std::vector<std::unique_ptr<brgemm_kernel_t>> brg_kernels_;
    std::vector<std::unique_ptr<jit_brgemm_kernel_post_ops>> kernels_po_;
    std::unique_ptr<jit_avx512_core_brgemm_conv_trans_kernel::
                    jit_avx512_core_brgemm_conv_trans_kernel_t>
            copy_to_pbuffer_;
    std::vector<S_t> brg_kernel_palettes_;

    const float *oscales;
    size_t acc_dsz, bia_dsz, src_dsz, wei_dsz, dst_dsz;

    const memory_desc_wrapper bias_d;

    // pre - calculated values
    std::vector<dim_t> owb_kw_top_vpads;
    std::vector<dim_t> owb_kw_bottom_vpads;

    int KD, KH, KW, EXT_KD, EXT_KH, EXT_KW, KS, KD_BLOCK, KH_BLOCK, KW_BLOCK,
            KD_BLOCK_PAD, KH_BLOCK_PAD, ID, IH, IW, IDP, IHP, IWP, OD, OH, OW,
            SD, SH, SW, FP, TP, LP, DD, DH, DW;
    dim_t src_w_sz, src_h_sz, src_d_sz, dst_w_sz, dst_h_sz, dst_d_sz, wei_ic_sz,
            wei_kw_sz, wei_kh_sz, wei_kd_sz, wei_ocb_sz;
    dim_t pbuf_w_sz, pbuf_h_sz, pbuf_d_sz;

    int ic_chunks;
    bool need_postwork;
    bool is_amx;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
