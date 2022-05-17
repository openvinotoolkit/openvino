/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_X64_GEMM_BF16_CONVOLUTION_HPP
#define CPU_X64_GEMM_BF16_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"
#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_convolution_utils.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/ref_depthwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <data_type_t dst_data_type>
struct gemm_bf16_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_fwd_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::bf16, data_type::bf16,
                            data_type::undef, dst_data_type, data_type::f32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type,
                                    data_type::bf16, data_type::f32))
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops,
                            dst_data_type)
                    && post_ops_ok();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads());
        }

        bool is_postprocess_required() const {
            bool post_ops_sum_only_for_dst_f32 = true
                    && dst_data_type == data_type::f32
                    && attr()->post_ops_.len() == 1
                    && attr()->post_ops_.contain(primitive_kind::sum, 0);
            bool is_pp_for_post_ops_required = true
                    && attr()->post_ops_.len() > 0
                    && !post_ops_sum_only_for_dst_f32;
            return dst_data_type == data_type::bf16 || with_bias()
                    || is_pp_for_post_ops_required;
        }

        conv_gemm_conf_t jcp_;

    protected:
        virtual bool post_ops_ok() const {
            auto const &po = this->attr()->post_ops_;
            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < po.len(); i++) {
                    ok = ok && utils::one_of(po.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise);
                }
                return ok;
            };

            auto contain = [&](dnnl::impl::primitive_kind_t kind) { return po.find(kind) != -1; };
            auto position = [&](dnnl::impl::primitive_kind_t kind) { return po.find(kind); };
            auto count = [&](dnnl::impl::primitive_kind_t kind) { return po.count(kind); };

            return all_post_ops_supported() &&
                   count(primitive_kind::sum) <= 1 &&
                   IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0);

            return false;
        }
    };

    gemm_bf16_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), pp_ker_(nullptr) {}

    typedef typename prec_traits<dst_data_type>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    status_t init(engine_t *engine) override {
        const auto &post_ops = pd()->attr()->post_ops_;
        const acc_data_t one = 1.0, zero = 0.0;
        beta_ = dst_data_type == data_type::f32
                        && post_ops.find(primitive_kind::sum) >= 0
                ? one
                : zero;

        if (this->pd()->is_postprocess_required()) {
            CHECK(safe_ptr_assign(pp_ker_, new pp_ker_t(this->pd())));
            return pp_ker_->create_kernel();
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        const bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_forward_nspc(ctx) : execute_forward_ncsp(ctx);
    }

private:
    status_t execute_forward_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_forward_nspc(const exec_ctx_t &ctx) const;
    status_t execute_forward_thr_nspc(const int ithr, const int nthr,
            const src_data_t *src_base, const wei_data_t *wei_base,
            const float *bia_base, dst_data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad,
            const void *post_ops_binary_rhs_arg_vec, int MB) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    class pp_ker_t : public jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(gemm_bf16_convolution_fwd_t::pp_kernel);
        pp_ker_t(const pd_t *pd);

        ~pp_ker_t() {
            for (auto inj : jit_eltwise_injectors_)
                delete inj;
            jit_eltwise_injectors_.clear();
        }

        void operator()(dst_data_t *dst, const acc_data_t *acc,
                const acc_data_t *bias, float sum_scale, size_t oc_work, size_t g_offset,
                const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
                const size_t g_oc_offset);
        void operator()(dst_data_t *dst, const acc_data_t *acc,
                const acc_data_t *bias,
                size_t g_offset, size_t start_oc, float sum_scale, size_t dst_str,
                size_t acc_str, size_t sp_len, size_t oc,
                const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
                const size_t g_oc_offset);

    private:
        struct ker_args {
            dst_data_t *dst;
            const acc_data_t *acc;
            const acc_data_t *bias;
            float sum_scale;
            size_t dst_stride_in_bytes;
            size_t acc_stride_in_bytes;
            size_t spatial_length;
            size_t oc_work;
            size_t oc_offset;

            size_t g_oc_offset;
            const void *post_ops_binary_rhs_arg_vec;
            const void *dst_orig;
        };

        enum { default_unroll_2_pow_ = 2 };

        Xbyak::Reg64 reg_param = rdi;
        Xbyak::Reg64 reg_dst_base = rdx;
        Xbyak::Reg64 reg_acc_base = rax;
        Xbyak::Reg64 reg_dst = rsi;
        Xbyak::Reg64 reg_acc = rbp;
        Xbyak::Reg64 reg_bias = rbx;

        Xbyak::Reg64 reg_len = r8;
        Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
        Xbyak::Reg64 reg_rem_mask = r9;
        Xbyak::Opmask kreg_rem_mask = k1;
        Xbyak::Reg64 reg_oc_iter = r11;
        Xbyak::Reg64 reg_len_iter = r12;
        Xbyak::Reg64 reg_dst_str = r13;
        Xbyak::Reg64 reg_acc_str = r14;

        using Vmm = typename cpu_isa_traits<avx512_common>::Vmm;
        Xbyak::Reg64 reg_oc_offset = r10;
        Xbyak::Reg64 reg_dw = r9;
        Xbyak::Reg64 reg_post_ops_data = reg_bias;
        Xbyak::Opmask kmask = k7;

        Xbyak::Reg64 reserved_eltwise_gpr = r10;
        Xbyak::Opmask reserved_eltwise_maskr = k2;

        Xbyak::Zmm vreg_sum_scale, vreg_bias, vreg_dw;

        Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(27);
        Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(28);
        Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(29);
        Xbyak::Reg64 bf16_emu_reserv_4 = r15;
        Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);
        Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(31);

        /* binary post-ops operands */
        Xbyak::Reg64 oc_off_oprnd = r12;
        Xbyak::Reg64 out_off_oprnd = r13;

        constexpr static int reg64_size = sizeof(int64_t);
        constexpr static int reg_binary_post_op_acc_off = 0;
        constexpr static int stack_space_needed = reg64_size;

        const conv_gemm_conf_t &jcp_;
        post_ops_t post_ops_;
        const bool do_sum_;
        int max_data_reg_idx_, max_unroll_, compute_reg_step_;
        int data_reg_base_idx_;
        size_t vlen_;
        cpu_isa_t isa_;
        std::unique_ptr<bf16_emulation_t> bf16_emu_;
        const primitive_attr_t* attr_;
        nstl::vector<jit_uni_eltwise_injector_f32<avx512_common>*> jit_eltwise_injectors_;

        void apply_postops(
                const bool apply_mask, const int out_offset, const int vmm_idx);
        void generate() override;
        int vreg_dst_idx(int iter) {
            int idx = data_reg_base_idx_ + iter * compute_reg_step_ + 0;
            assert(idx <= max_data_reg_idx_);
            return idx;
        }
        int vreg_prev_dst_idx(int iter) {
            int idx = data_reg_base_idx_ + iter * compute_reg_step_ + 1;
            assert(idx <= max_data_reg_idx_);
            return idx;
        }

        Xbyak::Zmm vreg_dst(int iter) {
            return Xbyak::Zmm(vreg_dst_idx(iter));
        };

        Xbyak::Ymm vreg_dst_ymm(int iter) {
            return Xbyak::Ymm(vreg_dst_idx(iter));
        };

        Xbyak::Zmm vreg_prev_dst(int iter) {
            return Xbyak::Zmm(vreg_prev_dst_idx(iter));
        };

        Xbyak::Ymm vreg_prev_dst_ymm(int iter) {
            return Xbyak::Ymm(vreg_prev_dst_idx(iter));
        };
    };

    acc_data_t beta_;
    std::unique_ptr<pp_ker_t> pp_ker_;
};

template <data_type_t diff_src_data_type>
struct gemm_bf16_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_bwd_data_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_data
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(diff_src_data_type, data_type::bf16,
                            data_type::undef, data_type::bf16, data_type::f32)
                    && !has_zero_dim_memory() && is_supported_post_ops();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md_, weights_md_, diff_dst_md_, bias_md_,
                    attr_, dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;


    protected:
        virtual bool is_supported_post_ops() const {
            const auto &p = this->attr()->post_ops_;
            if (p.len() > 1)
                return false;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len(); i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::depthwise);
                }
                return ok;
            };

            return all_post_ops_supported();
        }
    };

    gemm_bf16_convolution_bwd_data_t(const pd_t* apd) : primitive_t(apd) {
        const auto& post_ops = pd()->attr()->post_ops_;
        for (int i = 0; i < post_ops.len(); i++) {
            auto& post_op = post_ops.entry_[i];
            if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(new ref_depthwise_scalar_fwd_t(post_op.depthwise.alg));
            }
        }
    }

    ~gemm_bf16_convolution_bwd_data_t() {
        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<diff_src_data_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        const bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_backward_data_nspc(ctx)
                       : execute_backward_data_ncsp(ctx);
    }

private:
    status_t execute_backward_data_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_nspc(const exec_ctx_t &ctx) const;
    status_t execute_backward_data_thr_nspc(const int ithr, const int nthr,
            diff_src_data_t *diff_src_base, const wei_data_t *wei_base,
            const diff_dst_data_t *diff_dst_base,
            const memory_tracking::grantor_t &scratchpad, int MB,
            const std::vector<const void *>& post_ops_binary_rhs_arg_vec) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;
};

template <data_type_t diff_wei_data_type>
struct gemm_bf16_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_bwd_weights_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && desc()->prop_kind == prop_kind::backward_weights
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(data_type::bf16, diff_wei_data_type,
                            data_type::undef, data_type::bf16, data_type::f32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->diff_bias_desc.data_type,
                                    data_type::bf16, data_type::f32))
                    && !has_zero_dim_memory() && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, diff_weights_md_, diff_dst_md_,
                    diff_bias_md_, attr_, dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_;
    };

    gemm_bf16_convolution_bwd_weights_t(const pd_t *apd)
        : primitive_t(apd), acc_ker_(nullptr) {}

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<diff_wei_data_type>::type diff_wei_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(
                acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
        return acc_ker_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        const bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_backward_weights_nspc(ctx)
                       : execute_backward_weights_ncsp(ctx);
    }

private:
    void bf16_bwd_weights_reduction_par_ncsp(int ithr_mb, int nthr_mb,
            const conv_gemm_conf_t &jcp, const acc_data_t *weights_reduce_base,
            diff_wei_data_t *weights_base) const;
    void bf16_bwd_weights_reduction_par_nspc(int ithr_mb, int nthr_mb,
            size_t g_start, size_t g_end, const conv_gemm_conf_t &jcp,
            const acc_data_t *weights_reduce_base,
            diff_wei_data_t *weights_base) const;

    status_t execute_backward_weights_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_backward_weights_nspc(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<cpu_accumulator_1d_t<data_type::f32>> acc_ker_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
