/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef GEMM_X8S8S32X_CONVOLUTION_HPP
#define GEMM_X8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "ref_eltwise.hpp"
#include "gemm_convolution_utils.hpp"

#include "gemm/gemm.hpp"
#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t dst_type>
struct _gemm_x8s8s32x_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_IMPL_STR,
                _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>);

        virtual status_t init() override {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind,
                        prop_kind::forward_training,
                        prop_kind::forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->src_desc.data_type == src_type
                && this->desc()->dst_desc.data_type == dst_type
                && this->desc()->weights_desc.data_type == s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->desc()->accum_data_type == data_type::s32
                && utils::one_of(this->src_pd_.desc()->format, nhwc, ndhwc)
                && this->src_pd_.desc()->format == this->dst_pd_.desc()->format
                && IMPLICATION(this->src_pd_.desc()->format == nhwc,
                        this->weights_pd_.desc()->format == (this->with_groups()
                                ? ((src_type == data_type::s8) ? hwigo_s8s8 : hwigo)
                                : ((src_type == data_type::s8) ? hwio_s8s8 : hwio)))
                && IMPLICATION(this->src_pd_.desc()->format == ndhwc,
                        this->weights_pd_.desc()->format == (this->with_groups()
                                ? ((src_type == data_type::s8) ? dhwigo_s8s8 : dhwigo)
                                : ((src_type == data_type::s8) ? dhwio_s8s8 : dhwio)))
                && this->is_gemm_conv_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *this->desc(), this->src_pd(), this->weights_pd(0),
                    this->dst_pd(), *this->attr(), mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const size_t ndims_sp = this->desc()->src_desc.ndims - 4;
            return (utils::pick(ndims_sp, nhwc, ndhwc));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const size_t ndims_sp = this->desc()->src_desc.ndims - 4;
            return this->with_groups()
                ? (src_type == data_type::s8) ? utils::pick(ndims_sp, hwigo_s8s8, dhwigo_s8s8)
                                              : utils::pick(ndims_sp, hwigo, dhwigo)
                : (src_type == data_type::s8) ? utils::pick(ndims_sp, hwio_s8s8, dhwio_s8s8)
                                              : utils::pick(ndims_sp, hwio, dhwio);
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(src_format()));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }

        virtual bool is_gemm_conv_format() const {
            const auto &p = this->attr()->post_ops_;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len_; i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                                             primitive_kind::quantization);
                }
                return ok;
            };

            return all_post_ops_supported();
        }
    };

    _gemm_x8s8s32x_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {
        if (mayiuse(avx512_common)) {
            pp_ker_ = new jit_pp_ker_t<avx512_common>(this->pd());
        } else if (mayiuse(avx2)) {
            pp_ker_ = new jit_pp_ker_t<avx2>(this->pd());
        } else if (mayiuse(sse42)) {
            pp_ker_ = new jit_pp_ker_t<sse42>(this->pd());
        } else {
            pp_ker_ = new ref_pp_ker_t(this->pd());
        }
    }
    ~_gemm_x8s8s32x_convolution_fwd_t() {
        delete pp_ker_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    void execute_forward() const;
    // XXX: this is throwaway code that will become unnecessary when we have a
    // sufficiently advanced igemm jit generator that supports quantization,
    // relu, and whatnot
    struct ker_args {
        dst_data_t *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        float sum_scale;
        float signed_scale;
        size_t len;
        size_t oc_offset;
        const float *weights_zp;
        const int32_t *weights_zp_compensation;
        size_t g_offset;
    };

    class uni_pp_ker_t {
    public:
        virtual ~uni_pp_ker_t() = default;

        void (*ker_)(const ker_args *args);

        virtual void operator()(dst_data_t *dst, acc_data_t *acc, const char *bias, const float *scales, float signed_scale,
                int g, size_t start, size_t end, float* weights_zp, int32_t* weights_zp_compensation) = 0;

        size_t dst_os_stride_;
    };

    template <cpu_isa_t isa>
    class jit_pp_ker_t : public uni_pp_ker_t, jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(
        _gemm_x8s8s32x_convolution_fwd_t::pp_kernel);
        jit_pp_ker_t(const pd_t *pd);
        ~jit_pp_ker_t() {
            for (auto inj : jit_eltwise_injectors_)
                delete inj;
            jit_eltwise_injectors_.clear();
            for (auto inj : jit_depthwise_injectors_)
                delete inj;
            jit_depthwise_injectors_.clear();
        }

        void operator()(dst_data_t *dst, acc_data_t *acc,
            const char *bias, const float *scales, float signed_scale,
            int g, size_t start, size_t end, float* weights_zp, int32_t* weights_zp_compensation) override;

    private:
        void generate();

        void(*ker_)(const ker_args *args);

        nstl::vector<jit_uni_eltwise_injector_f32<isa> *> jit_eltwise_injectors_;
        nstl::vector<jit_uni_depthwise_injector_f32<isa> *> jit_depthwise_injectors_;

        using Vmm = typename cpu_isa_traits<isa>::Vmm;
        static const size_t vlen = cpu_isa_traits<isa>::vlen / sizeof(float);

        Xbyak::Reg64 reg_param = abi_param1;
        Xbyak::Reg64 reg_dst = rdx;
        Xbyak::Reg64 reg_acc = rax;
        Xbyak::Reg64 reg_bias = rbx;
        Xbyak::Reg64 reg_scales = rsi;
        Xbyak::Reg64 reg_g_offset = rbp;

        Xbyak::Reg64 reg_len = r8;
        Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
        Xbyak::Reg64 reg_oc_offset = r9;
        Xbyak::Reg64 reg_rem_mask_short = r10;
        Xbyak::Opmask kreg_rem_mask_short = k1;

        Vmm vreg_zero, vreg_scale, vreg_sum_scale, vreg_signed_scale, vreg_comp;

        //  sse42/avx2
        Xbyak::Reg64 reg_ptr_maskmovdqu_dst = rdi; // sse42: store destination - must be rdi
        Xbyak::Reg8 reg_tmp_8 = r11b;
        Xbyak::Reg32 reg_tmp_32 = r11d;
        Xbyak::Reg64 reg_tmp_64 = r11;
        Xbyak::Label l_table;
        Xbyak::Reg64 reg_table = r12;
        Xbyak::Reg64 reg_shift_table = r13;
        Vmm vreg_mask = Vmm(0); //  sse42: mask for blendvps must be in xmm0
        Vmm vreg_store_mask = Vmm(1);

        //  post_ops
        Xbyak::Opmask mask_post_op_reserved = k2;
        Xbyak::Reg64 eltwise_reserved = rax;
        Xbyak::Reg64 reg_d_weights = r14;
        Xbyak::Reg64 reg_d_bias = r15;
        Vmm vreg_d_weights, vreg_d_bias;
        post_ops_t post_ops_;

        Xbyak::Reg64 reg_weights_zp = r14;
        Xbyak::Reg64 reg_weights_zp_compensation = r15;

        const jit_gemm_conv_conf_t jcp_;
        size_t OC_;
        size_t OS_;
        data_type_t bias_data_type_;
        size_t bias_data_type_size_;
        bool do_scale_;
        size_t scale_idx_mult_;
        round_mode_t rmode_;
        bool do_bias_;
        bool do_eltwise_;
        bool do_sum_;
        float sum_scale_;
        data_type_t sum_data_type_;
        bool do_signed_scaling_;
        bool with_weights_zp_;

        int default_OC_loop_unroll_;
        int max_OC_loop_unroll_;
        int idx_compute_vreg_start_;
        int idx_compute_vreg_max_;
        int compute_vregs_per_iter_;

        int idx_vreg_dst(int iter) {
            int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 0;
            assert(idx <= idx_compute_vreg_max_);
            return idx;
        }
        int idx_vreg_bias(int iter) {
            int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 1;
            assert(idx <= idx_compute_vreg_max_);
            return idx;
        }
        int idx_vreg_prev_dst(int iter) {
            int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 2;
            assert(idx <= idx_compute_vreg_max_);
            return idx;
        }


        Vmm vreg_dst(int idx) { return Vmm(idx_vreg_dst(idx)); };
        Xbyak::Ymm ymm_dst(int idx) { return Xbyak::Ymm(idx_vreg_dst(idx)); };
        Xbyak::Xmm xmm_dst(int idx) { return Xbyak::Xmm(idx_vreg_dst(idx)); };
        Vmm vreg_bias(int idx) { return Vmm(idx_vreg_bias(idx)); };
        Vmm vreg_prev_dst(int idx) { return Vmm(idx_vreg_prev_dst(idx)); };
    };

    class ref_pp_ker_t : public uni_pp_ker_t {
    public:
        ref_pp_ker_t(const pd_t *pd);
        ~ref_pp_ker_t() {
            for (auto impl : ref_eltwise_injectors_)
                delete impl;
            ref_eltwise_injectors_.clear();
            for (auto impl : ref_depthwise_injectors_)
                delete impl;
            ref_depthwise_injectors_.clear();
        }

        void operator()(dst_data_t *dst, acc_data_t *acc,
                        const char *bias, const float *scales, float signed_scale,
                        int g, size_t start, size_t end, float* weights_zp, int32_t* weights_zp_compensation) override;

    private:
        void generate();

        void(*ker_)(const ker_args *args);

        nstl::vector<ref_eltwise_scalar_fwd_t*> ref_eltwise_injectors_;
        nstl::vector<ref_depthwise_scalar_fwd_t*> ref_depthwise_injectors_;

        post_ops_t post_ops_;
        const jit_gemm_conv_conf_t jcp_;
        size_t OC_;
        data_type_t bias_data_type_;
        bool do_scale_;
        size_t scale_idx_mult_;
        round_mode_t rmode_;
        bool do_bias_;
        bool with_weights_zp_;
    };

    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src_base, const wei_data_t *wei_base,
            const char *bia_base, dst_data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;

    int nthr_;
    uni_pp_ker_t *pp_ker_;
};

template <data_type_t dst_type>
struct _gemm_u8s8s32x_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t{
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_IMPL_STR,
                _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>);

        virtual status_t init() override {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == prop_kind::backward_data
                && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->diff_src_desc.data_type == dst_type
                && this->desc()->diff_dst_desc.data_type == u8
                && this->desc()->weights_desc.data_type == s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->desc()->accum_data_type == data_type::s32
                && utils::everyone_is(nhwc, this->diff_src_pd_.desc()->format,
                        this->diff_dst_pd_.desc()->format)
                && this->weights_pd_.desc()->format == (this->with_groups()
                        ? hwigo : hwio)
                && attr()->post_ops_.has_default_values();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *this->desc(), this->diff_src_pd(), this->weights_pd(0),
                    this->diff_dst_pd(), *this->attr(), mkldnn_get_max_threads());
        }

        virtual bool support_bias() const override { return true; }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nhwc));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nhwc));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(
                            this->with_groups() ? hwigo : hwio));
            if (bias_pd_.desc()->format == any)
                CHECK(bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
             return status::success;
        }
    };

    _gemm_u8s8s32x_convolution_bwd_data_t(const pd_t *apd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {}
    ~_gemm_u8s8s32x_convolution_bwd_data_t() {}

    typedef typename prec_traits<data_type::u8>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) const {
        execute_backward_data();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    void execute_backward_data_thr(const int ithr, const int nthr,
            const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
            const char *bia_base, diff_src_data_t *diff_src_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
