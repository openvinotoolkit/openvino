/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef CPU_JIT_UNI_DW_CONVOLUTION_HPP
#define CPU_JIT_UNI_DW_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_barrier.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_reducer.hpp"

#include "jit_uni_dw_conv_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct _jit_uni_dw_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_fwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->dst_desc.data_type)
                && IMPLICATION(this->with_bias(),
                        data_type::f32 == this->desc()->bias_desc.data_type);

            if (!ok) return status::unimplemented;

            status_t status = jit_uni_dw_conv_fwd_kernel_f32<isa>::init_conf(
                    jcp_, *this->desc(), this->src_pd_.desc(),
                    *this->weights_pd_.desc(), *this->dst_pd_.desc(),
                    *this->attr());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_fwd_kernel_f32<isa>::init_scratchpad(scratchpad,
                    jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
            auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(desired_act_fmt));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(desired_act_fmt));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(desired_wei_fmt));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    _jit_uni_dw_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr)
    { kernel_ = new jit_uni_dw_conv_fwd_kernel_f32<isa>(pd()->jcp_, *pd()->attr()); }

    ~_jit_uni_dw_convolution_fwd_t() { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_uni_dw_conv_fwd_kernel_f32<isa> *kernel_;
};

using jit_avx512_common_dw_convolution_fwd_t =
    _jit_uni_dw_convolution_fwd_t<avx512_common>;
using jit_avx2_dw_convolution_fwd_t = _jit_uni_dw_convolution_fwd_t<avx2>;
using jit_sse42_dw_convolution_fwd_t = _jit_uni_dw_convolution_fwd_t<sse42>;

template <cpu_isa_t isa>
struct _jit_uni_dw_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;

            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, backward,
                        backward_data)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->diff_src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type);

            if (!ok) return status::unimplemented;

            status_t status =
                jit_uni_dw_conv_bwd_data_kernel_f32<isa>::init_conf(jcp_,
                        *this->desc(), *this->diff_src_pd_.desc(),
                        *this->weights_pd_.desc(), *this->diff_dst_pd_.desc());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_bwd_data_kernel_f32<isa>::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
            auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(desired_act_fmt));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(desired_act_fmt));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(desired_wei_fmt));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));

            return status::success;
        }
    };

    _jit_uni_dw_convolution_bwd_data_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
    { kernel_ = new jit_uni_dw_conv_bwd_data_kernel_f32<isa>(pd()->jcp_); }
    ~_jit_uni_dw_convolution_bwd_data_t() { delete kernel_; };

    typedef typename prec_traits<data_type::f32>::type data_t;

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

    jit_uni_dw_conv_bwd_data_kernel_f32<isa> *kernel_;
};

using jit_avx512_common_dw_convolution_bwd_data_t =
    _jit_uni_dw_convolution_bwd_data_t<avx512_common>;
using jit_avx2_dw_convolution_bwd_data_t =
    _jit_uni_dw_convolution_bwd_data_t<avx2>;
using jit_sse42_dw_convolution_bwd_data_t =
    _jit_uni_dw_convolution_bwd_data_t<sse42>;

template <cpu_isa_t isa>
struct _jit_uni_dw_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_bwd_weights_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;

            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == prop_kind::backward_weights
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->diff_weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type);

            if (!ok) return status::unimplemented;

            const int max_threads = mkldnn_in_parallel()
                ? 1 : mkldnn_get_max_threads();

            status_t status =
                jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::init_conf(jcp_,
                        *this->desc(), *this->src_pd_.desc(),
                        *this->diff_weights_pd_.desc(),
                        *this->diff_dst_pd_.desc(), max_threads);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_bwd_weights_kernel_f32<isa>::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt = isa == avx512_common ? nChw16c : nChw8c;
            auto desired_wei_fmt = isa == avx512_common ? Goihw16g : Goihw8g;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(desired_act_fmt));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(desired_act_fmt));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(desired_wei_fmt));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));

            return status::success;
        }
    };

    _jit_uni_dw_convolution_bwd_weights_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);

    ~_jit_uni_dw_convolution_bwd_weights_t() {
        delete kernel_;
        delete acc_ker_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights() const;
    bool do_parallel_reduction() const { return false; }
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_uni_dw_conv_bwd_weights_kernel_f32<isa> *kernel_;
    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
};

using jit_avx512_common_dw_convolution_bwd_weights_t =
    _jit_uni_dw_convolution_bwd_weights_t<avx512_common>;
using jit_avx2_dw_convolution_bwd_weights_t =
    _jit_uni_dw_convolution_bwd_weights_t<avx2>;
using jit_sse42_dw_convolution_bwd_weights_t =
    _jit_uni_dw_convolution_bwd_weights_t<sse42>;

}
}
}

#endif
