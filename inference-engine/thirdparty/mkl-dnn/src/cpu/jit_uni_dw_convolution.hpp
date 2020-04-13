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

#ifndef CPU_JIT_UNI_DW_CONVOLUTION_HPP
#define CPU_JIT_UNI_DW_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_barrier.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_reducer.hpp"

#include "jit_uni_dw_conv_kernel_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, data_type_t src_type, data_type_t dst_type = src_type>
struct _jit_uni_dw_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_fwd_t<isa, src_type, dst_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                               forward_inference)
                    && utils::one_of(this->desc()->alg_kind,
                               alg_kind::convolution_auto,
                               alg_kind::convolution_direct)
                    && !this->has_zero_dim_memory()
                    && utils::everyone_is(src_type,
                               this->desc()->src_desc.data_type,
                               this->desc()->weights_desc.data_type)
                    && this->desc()->dst_desc.data_type == dst_type
                    && IMPLICATION(this->with_bias(), utils::one_of(
                        this->desc()->bias_desc.data_type, data_type::f32,
                        data_type::bf16))
                    && !this->attr()->has_asymmetric_quantization();
            if (!ok)
                return status::unimplemented;

            status_t status
                    = jit_uni_dw_conv_fwd_kernel<isa, src_type>::init_conf(jcp_,
                            *this->desc(), this->src_pd_.desc(),
                            *this->weights_pd_.desc(), *this->dst_pd_.desc(),
                            *this->attr());
            if (status != status::success)
                return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_fwd_kernel<isa, src_type>::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt = ndims() == 5 ? utils::one_of(isa, avx512_common, avx512_core) ? nCdhw16c : nCdhw8c
                                                : utils::one_of(isa, avx512_common, avx512_core) ? nChw16c : nChw8c;
            auto desired_wei_fmt = ndims() == 5 ? utils::one_of(isa, avx512_common, avx512_core) ? Goidhw16g : Goidhw8g
                                                : utils::one_of(isa, avx512_common, avx512_core) ? Goihw16g : Goihw8g;

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
        : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr) {
        kernel_ = new jit_uni_dw_conv_fwd_kernel<isa, src_type>(pd()->jcp_, *pd()->attr());
    }

    ~_jit_uni_dw_convolution_fwd_t() { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type f32_data_t;
    typedef typename prec_traits<data_type::bf16>::type bf16_data_t;
    typedef typename prec_traits<src_type>::type data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_uni_dw_conv_fwd_kernel<isa, src_type> *kernel_;
};

using jit_avx512_common_dw_convolution_fwd_t
        = _jit_uni_dw_convolution_fwd_t<avx512_common, data_type::f32>;
using jit_avx2_dw_convolution_fwd_t
        = _jit_uni_dw_convolution_fwd_t<avx2, data_type::f32>;
using jit_sse42_dw_convolution_fwd_t
        = _jit_uni_dw_convolution_fwd_t<sse42, data_type::f32>;

template <cpu_isa_t isa, data_type_t diff_dst_type,
        data_type_t diff_src_type = diff_dst_type>
struct _jit_uni_dw_convolution_bwd_data_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_bwd_data_t<isa, diff_dst_type,
                                    diff_src_type>);

        virtual status_t init() override {
            using namespace prop_kind;

            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && utils::one_of(
                               this->desc()->prop_kind, backward, backward_data)
                    && utils::one_of(this->desc()->alg_kind,
                               alg_kind::convolution_auto,
                               alg_kind::convolution_direct)
                    && !this->has_zero_dim_memory()
                    && utils::everyone_is(diff_dst_type,
                               this->desc()->weights_desc.data_type,
                               this->desc()->diff_dst_desc.data_type)
                    && diff_src_type == this->desc()->diff_src_desc.data_type;

            if (!ok)
                return status::unimplemented;

            status_t status = jit_uni_dw_conv_bwd_data_kernel<isa,
                    diff_dst_type>::init_conf(jcp_, *this->desc(),
                    *this->diff_src_pd_.desc(), *this->weights_pd_.desc(),
                    *this->diff_dst_pd_.desc(), *this->attr());
            if (status != status::success)
                return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_bwd_data_kernel<isa,
                    diff_dst_type>::init_scratchpad(scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt
                    = utils::one_of(isa, avx512_common, avx512_core) ? nChw16c
                                                                     : nChw8c;
            auto desired_wei_fmt
                    = utils::one_of(isa, avx512_common, avx512_core) ? Goihw16g
                                                                     : Goihw8g;

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
        : cpu_primitive_t(apd, inputs, outputs) {
        kernel_ = new jit_uni_dw_conv_bwd_data_kernel<isa, diff_dst_type>(
                pd()->jcp_, *pd()->attr());
    }
    ~_jit_uni_dw_convolution_bwd_data_t() { delete kernel_; };

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_wei_data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_data: execute_backward_data(); break;
        default: assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_uni_dw_conv_bwd_data_kernel<isa, diff_dst_type> *kernel_;
};

using jit_avx512_common_dw_convolution_bwd_data_t
        = _jit_uni_dw_convolution_bwd_data_t<avx512_common, data_type::f32>;
using jit_avx2_dw_convolution_bwd_data_t
        = _jit_uni_dw_convolution_bwd_data_t<avx2, data_type::f32>;
using jit_sse42_dw_convolution_bwd_data_t
        = _jit_uni_dw_convolution_bwd_data_t<sse42, data_type::f32>;

template <cpu_isa_t isa, data_type_t src_type,
        data_type_t diff_weights_type = src_type>
struct _jit_uni_dw_convolution_bwd_weights_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_dw:", isa, ""),
                _jit_uni_dw_convolution_bwd_weights_t<isa, src_type,
                                    diff_weights_type>);

        virtual status_t init() override {
            using namespace prop_kind;

            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && this->set_default_params() == status::success
                    && this->desc()->prop_kind == prop_kind::backward_weights
                    && utils::one_of(this->desc()->alg_kind,
                               alg_kind::convolution_auto,
                               alg_kind::convolution_direct)
                    && utils::everyone_is(src_type,
                               this->desc()->src_desc.data_type,
                               this->desc()->diff_dst_desc.data_type)
                    && this->desc()->diff_weights_desc.data_type
                            == diff_weights_type;

            if (!ok)
                return status::unimplemented;

            const int max_threads
                    = mkldnn_in_parallel() ? 1 : mkldnn_get_max_threads();

            status_t status = jit_uni_dw_conv_bwd_weights_kernel<isa,
                    src_type>::init_conf(jcp_, *this->desc(),
                    *this->src_pd_.desc(), *this->diff_weights_pd_.desc(),
                    *this->diff_dst_pd_.desc(), max_threads);
            if (status != status::success)
                return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_dw_conv_bwd_weights_kernel<isa, src_type>::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            auto desired_act_fmt
                    = utils::one_of(isa, avx512_common, avx512_core) ? nChw16c
                                                                     : nChw8c;
            auto desired_wei_fmt
                    = utils::one_of(isa, avx512_common, avx512_core) ? Goihw16g
                                                                     : Goihw8g;

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
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , acc_ker_(nullptr)
        , kernel_(nullptr) {
        kernel_ = new jit_uni_dw_conv_bwd_weights_kernel<isa, src_type>(
                pd()->jcp_);

        if (pd()->jcp_.nthr_mb > 1 && isa != sse42)
            acc_ker_ = new cpu_accumulator_1d_t<data_type::f32>();
    }

    ~_jit_uni_dw_convolution_bwd_weights_t() {
        delete acc_ker_;
        delete kernel_;
    };

    typedef typename prec_traits<data_type::f32>::type f32_data_t;
    typedef typename prec_traits<data_type::bf16>::type bf16_data_t;
    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<src_type>::type diff_dst_data_t;
    typedef typename prec_traits<diff_weights_type>::type diff_weights_data_t;

    virtual void execute(event_t *e) const {
        execute_backward_weights();
        execute_reduction();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights() const;
    void execute_reduction() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
    jit_uni_dw_conv_bwd_weights_kernel<isa, src_type> *kernel_;
};

using jit_avx512_common_dw_convolution_bwd_weights_t
        = _jit_uni_dw_convolution_bwd_weights_t<avx512_common, data_type::f32>;
using jit_avx2_dw_convolution_bwd_weights_t
        = _jit_uni_dw_convolution_bwd_weights_t<avx2, data_type::f32>;
using jit_sse42_dw_convolution_bwd_weights_t
        = _jit_uni_dw_convolution_bwd_weights_t<sse42, data_type::f32>;

}
}
}

#endif
