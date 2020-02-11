/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_JIT_GEMM_BF16_CONVOLUTION_HPP
#define CPU_JIT_GEMM_BF16_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "gemm_convolution_utils.hpp"
#include "gemm/gemm.hpp"
#include "jit_avx512_core_bf16cvt.hpp"
#include "jit_uni_eltwise.hpp"
#include "cpu_reducer.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t dst_data_type>
struct gemm_bf16_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                           forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::bf16,
                           this->desc()->src_desc.data_type,
                           this->desc()->weights_desc.data_type)
                && dst_data_type == this->desc()->dst_desc.data_type
                && this->src_pd_.desc()->format == src_format()
                && this->dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format()
                && this->is_gemm_conv_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_pd(), weights_pd(0), dst_pd(), *attr(),
                    mkldnn_get_max_threads());
        }

        bool is_postprocess_required() const {
            bool post_ops_sum_only_for_dst_f32 = true
                && dst_data_type == data_type::f32
                && attr()->post_ops_.len_ == 1
                && attr()->post_ops_.contain(primitive_kind::sum, 0);
            bool is_pp_for_post_ops_required = true
                && attr()->post_ops_.len_ > 0
                && !post_ops_sum_only_for_dst_f32;
            return dst_data_type == data_type::bf16
                       || with_bias()
                       || is_pp_for_post_ops_required;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
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
            auto const &po = this->attr()->post_ops_;
            auto is_eltwise = [&](int idx)
            { return po.entry_[idx].is_eltwise(); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };

            switch (po.len_) {
            case 0: return true; // no post_ops
            case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
            case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
            default: return false;
            }
        }
    };

    gemm_bf16_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true), pp_ker_(nullptr)
    {
        const auto &post_ops = pd()->attr()->post_ops_;
        const acc_data_t one = 1.0, zero = 0.0;
        beta_ = dst_data_type == data_type::f32
            && post_ops.find(primitive_kind::sum) >= 0
                ? one
                : zero;

        if (this->pd()->is_postprocess_required())
            pp_ker_ = new pp_ker_t(this->pd());
    }

    ~gemm_bf16_convolution_fwd_t() {
        delete pp_ker_;
    }

    typedef typename prec_traits<dst_data_type>::type dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    class pp_ker_t : jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(
        gemm_bf16_convolution_fwd_t::pp_kernel);
        pp_ker_t(const pd_t *pd);

        ~pp_ker_t() {
            delete bf16_emu_;
            delete eltwise_injector_;
        }

        void operator()(dst_data_t *dst, const acc_data_t *acc,
            const acc_data_t *bias, float sum_scale,
            size_t dst_str, size_t acc_str, size_t len, bool do_parallel);

        size_t dst_os_stride_;

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
        };

        enum {
            default_unroll_2_pow_ = 2
        };

        Xbyak::Reg64 reg_param = abi_param1;
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

        Xbyak::Reg64 reserved_eltwise_gpr = r10;
        Xbyak::Opmask reserved_eltwise_maskr = k2;

        Xbyak::Zmm vreg_sum_scale, vreg_bias;

        Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(27);
        Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(28);
        Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(29);
        Xbyak::Reg64 bf16_emu_reserv_4 = r11;
        Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(30);
        Xbyak::Zmm bf16_emu_reserv_6 = Xbyak::Zmm(31);

        void(*ker_)(const ker_args *args);
        const jit_gemm_conv_conf_t &jcp_;
        size_t OC_;
        bool do_bias_;
        bool do_eltwise_;
        bool do_sum_;
        int max_data_reg_idx_, max_unroll_, compute_reg_step_;
        int data_reg_base_idx_;
        size_t vlen_;
        bool is_cpx_;
        bf16_emulation_t *bf16_emu_;
        jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

        void generate();
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
    pp_ker_t *pp_ker_;
};

template <data_type_t diff_src_data_type>
struct gemm_bf16_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::bf16,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type)
                && diff_src_data_type == this->desc()->diff_src_desc.data_type
                && this->diff_src_pd_.desc()->format == src_format()
                && this->diff_dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_pd(), weights_pd(0), diff_dst_pd(), *attr(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->diff_src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->diff_src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    gemm_bf16_convolution_bwd_data_t(const pd_t *apd,
              const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {}

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<diff_src_data_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <data_type_t diff_wei_data_type>
struct gemm_bf16_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_bf16_convolution_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
            && this->set_default_params() == status::success
            && this->desc()->prop_kind == backward_weights
            && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                       alg_kind::convolution_direct)
            && !this->has_zero_dim_memory()
            && utils::everyone_is(data_type::bf16,
                    this->desc()->src_desc.data_type,
                    this->desc()->diff_dst_desc.data_type)
            && diff_wei_data_type == this->desc()->diff_weights_desc.data_type
            && this->src_pd_.desc()->format == src_format()
            && this->diff_dst_pd_.desc()->format == src_format()
            && this->diff_weights_pd_.desc()->format == wei_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_pd(), diff_weights_pd(0), diff_dst_pd(), *attr(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(wei_format()));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    gemm_bf16_convolution_bwd_weights_t(const pd_t *apd,
              const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true)
        , acc_ker_(nullptr)
    {
        acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();
    }
    ~gemm_bf16_convolution_bwd_weights_t() {
        delete acc_ker_;
    }

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::f32>::type acc_data_t;
    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<diff_wei_data_type>::type diff_wei_data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_weights:
            execute_backward_weights();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void bf16_bwd_weights_reduction_par(int ithr_mb, int nthr_mb,
        const jit_gemm_conv_conf_t &jcp, const acc_data_t *weights_reduce_base,
        diff_wei_data_t *weights_base) const;

    void execute_backward_weights() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
};

}
}
}

#endif
