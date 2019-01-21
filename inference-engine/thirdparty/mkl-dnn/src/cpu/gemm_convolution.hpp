/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_JIT_GEMM_CONVOLUTION_HPP
#define CPU_JIT_GEMM_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "gemm_convolution_utils.hpp"
#include "gemm/gemm.hpp"
#include "scratchpad.hpp"
#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu>
struct _gemm_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine,
                const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, _gemm_convolution_fwd_t<with_relu>);

        inline memory_format_t src_format()
        {
            using namespace memory_format;
            return (utils::pick(this->cdesc_().src_desc.ndims - 3,
                ncw, nchw, ncdhw));
        }
        inline memory_format_t wei_format()
        {
            using namespace memory_format;
            return (this->with_groups()
                ? utils::pick(this->cdesc_().src_desc.ndims - 3,
                    goiw, goihw, goidhw)
                : utils::pick(this->cdesc_().src_desc.ndims - 3,
                    oiw, oihw, oidhw));
        }

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                           forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                           this->cdesc_().src_desc.data_type,
                           this->cdesc_().weights_desc.data_type,
                           this->cdesc_().dst_desc.data_type)
                && IMPLICATION(this->with_bias(), data_type::f32
                                   == this->cdesc_().bias_desc.data_type)
                && this->src_pd_.desc()->format == src_format()
                && this->dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format()
                && this->is_gemm_conv_format();
            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
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
            return status::success;
        }

        virtual bool is_gemm_conv_format() const {
            bool ok = true;
            auto const &po = this->attr()->post_ops_;

            auto is_eltwise = [&](int idx) { return po.entry_[idx].is_eltwise(); };
            auto is_depthwise = [&](int idx) { return po.entry_[idx].is_depthwise(); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };
            auto is_simple = [&](int idx) { return (is_eltwise(idx) || is_depthwise(idx)); };

            switch (po.len_) {
                using namespace mkldnn::impl::primitive_kind;
            case 0: // no post_ops
                break;
            case 1:
                ok = ok && // sum OR eltwise/depthwise
                        (is_simple(0) || is_sum(0));
                break;
            case 2:
                ok = ok && // sum->eltwise/depthwise OR eltwise/depthwise->eltwise/depthwise
                           ((is_sum(0) && is_simple(1)) || (is_simple(0) && is_simple(1)));
                break;
            case 3:
                ok = ok && // sum->eltwise/depthwise->eltwise/depthwise
                     (is_sum(0) && is_simple(1) && is_simple(2));
                break;

            default: ok = false;
            }
            return ok;
        }
    };

    _gemm_convolution_fwd_t(const pd_t *pd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , scratchpad_(nullptr)
    {
        using namespace prop_kind;

        const auto &post_ops = conf_.attr()->post_ops_;
        const data_t one = 1.0, zero = 0.0;
        beta_ = post_ops.find(primitive_kind::sum) >= 0 ? one : zero;

        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *(conf_.cdesc()), conf_.src_pd(), conf_.weights_pd(0),
            conf_.dst_pd(), mkldnn_get_max_threads(), with_relu,
            conf_.negative_slope());

        size_t size = (size_t)conf_.jcp_.im2col_sz * sizeof(data_t);
        jit_gemm_convolution_utils::prepare_scratchpad(this->conf_.jcp_,
                &this->scratchpad_, size, this->conf_.jcp_.nthr);

        for (int i = 0; i < post_ops.len_; i++) {
            auto &post_op = post_ops.entry_[i];
            if (post_op.is_eltwise()) {
                eltwise_injectors.push_back(new ref_eltwise_scalar_fwd_t(
                        post_op.eltwise.alg,
                        post_op.eltwise.alpha,
                        post_op.eltwise.beta
                ));
            } else if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(new ref_depthwise_scalar_fwd_t(
                        post_op.depthwise.alg
                ));
            }
        }

        use_fast_relu = false;
        if (conf_.jcp_.with_relu && post_ops.len_ == 0) {
            use_fast_relu = true;
            fast_relu_ns = conf_.jcp_.relu_negative_slope;
        } else if (post_ops.len_ == 1 && post_ops.entry_[0].is_relu(true, false)) {
            use_fast_relu = true;
            fast_relu_ns = post_ops.entry_[0].eltwise.alpha;
        } else if (post_ops.len_ == 2 && post_ops.entry_[0].is_sum() && post_ops.entry_[1].is_relu(true, false)) {
            use_fast_relu = true;
            fast_relu_ns = post_ops.entry_[1].eltwise.alpha;
        }
    }

    ~_gemm_convolution_fwd_t() {
        delete this->scratchpad_;

        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    scratchpad_t *scratchpad_;
    data_t beta_;

    nstl::vector<ref_eltwise_scalar_fwd_t*> eltwise_injectors;
    nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;

    bool use_fast_relu;
    float fast_relu_ns;
};

using gemm_convolution_fwd_t =
                         _gemm_convolution_fwd_t<false>;
using gemm_convolution_relu_t =
                         _gemm_convolution_fwd_t<true>;

struct gemm_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_data_t);

        inline memory_format_t src_format()
        {
            using namespace memory_format;
            return (utils::pick(this->desc()->diff_src_desc.ndims - 3,
                ncw, nchw, ncdhw));
        }
        inline memory_format_t wei_format()
        {
            using namespace memory_format;
            return (this->with_groups()
                ? utils::pick(this->desc()->diff_src_desc.ndims - 3,
                    goiw, goihw, goidhw)
                : utils::pick(this->desc()->diff_src_desc.ndims - 3,
                    oiw, oihw, oidhw));
        }

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->diff_src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type)
                && this->diff_src_pd_.desc()->format == src_format()
                && this->diff_dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format();
            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            return status::success;
        }
    };

    gemm_convolution_bwd_data_t(const pd_t *pd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , scratchpad_(nullptr)
    {
        using namespace prop_kind;

        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *(conf_.desc()), conf_.diff_src_pd(), conf_.weights_pd(0),
            conf_.diff_dst_pd(), mkldnn_get_max_threads());

        size_t size = (size_t)conf_.jcp_.im2col_sz * sizeof(data_t);
        jit_gemm_convolution_utils::prepare_scratchpad(this->conf_.jcp_,
                &this->scratchpad_, size, this->conf_.jcp_.nthr);
    }

    ~gemm_convolution_bwd_data_t() {
        delete this->scratchpad_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data();
    pd_t conf_;
    scratchpad_t *scratchpad_;
};

struct gemm_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_weights_t);

        inline memory_format_t src_format()
        {
            using namespace memory_format;
            return (utils::pick(this->desc()->src_desc.ndims - 3,
                ncw, nchw, ncdhw));
        }
        inline memory_format_t wei_format()
        {
            using namespace memory_format;
            return (this->with_groups()
                ? utils::pick(this->desc()->src_desc.ndims - 3,
                    goiw, goihw, goidhw)
                : utils::pick(this->desc()->src_desc.ndims - 3,
                    oiw, oihw, oidhw));
        }

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
            && this->set_default_params() == status::success
            && this->desc()->prop_kind == backward_weights
            && this->desc()->alg_kind == alg_kind::convolution_direct
            && !this->has_zero_dim_memory()
            && utils::everyone_is(data_type::f32,
                    this->desc()->src_desc.data_type,
                    this->desc()->diff_weights_desc.data_type,
                    this->desc()->diff_dst_desc.data_type)
            && IMPLICATION(this->with_bias(),
                    data_type::f32 == this->desc()->diff_bias_desc.data_type)
            && this->src_pd_.desc()->format == src_format()
            && this->diff_dst_pd_.desc()->format == src_format()
            && this->diff_weights_pd_.desc()->format == wei_format();
            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
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
            return status::success;
        }
    };

    gemm_convolution_bwd_weights_t(const pd_t *pd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , scratchpad_(nullptr)
    {
        using namespace prop_kind;

        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *(conf_.desc()), conf_.src_pd(), conf_.diff_weights_pd(0),
            conf_.diff_dst_pd(), mkldnn_get_max_threads());
        const memory_desc_wrapper weights_d(conf_.diff_weights_pd(0));

        size_t size = (size_t)conf_.jcp_.im2col_sz  * sizeof(data_t);
        if (conf_.jcp_.need_wei_reduction)
            size += (size_t)conf_.jcp_.ngroups * weights_d.size();

        jit_gemm_convolution_utils::prepare_scratchpad(this->conf_.jcp_,
                &this->scratchpad_, size, conf_.jcp_.nthr);
    }

    ~gemm_convolution_bwd_weights_t() {
        delete this->scratchpad_;
     };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        switch (conf_.desc()->prop_kind) {
        case prop_kind::backward_weights:
            execute_backward_weights();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights();
    pd_t conf_;
    scratchpad_t *scratchpad_;
};

}
}
}

#endif
