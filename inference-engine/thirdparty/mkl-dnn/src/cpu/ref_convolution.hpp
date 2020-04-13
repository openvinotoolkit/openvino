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

#ifndef CPU_REF_CONVOLUTION_HPP
#define CPU_REF_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type,
         impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type,
         impl::data_type_t acc_type = dst_type>
struct ref_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && this->desc()->src_desc.data_type == src_type
                && this->desc()->weights_desc.data_type == wei_type
                && this->desc()->accum_data_type == acc_type
                && this->desc()->dst_desc.data_type == dst_type
                && IMPLICATION(this->with_bias(), true
                        && IMPLICATION(src_type == u8,
                            utils::one_of(this->desc()->bias_desc.data_type,
                                f32, s32, s8, u8))
                        && IMPLICATION(src_type == f32,
                            this->desc()->bias_desc.data_type == f32))
                && is_supported_post_ops();
            return ok ? status::success : status::unimplemented;
        }

        virtual bool is_supported_post_ops() const {
            const auto &p = this->attr()->post_ops_;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len_; i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                                             primitive_kind::quantization);
                }
                return ok;
            };
            auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind); };

            return all_post_ops_supported() &&
                   count(primitive_kind::sum) <= 1;
        }
    };

    ref_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
        const auto &post_ops = pd()->attr()->post_ops_;

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
    }

    ~ref_convolution_fwd_t() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            execute_forward();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    nstl::vector<ref_eltwise_scalar_fwd_t*> eltwise_injectors;
    nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;
};

template <impl::data_type_t diff_src_type, impl::data_type_t wei_type,
         impl::data_type_t diff_dst_type,
         impl::data_type_t acc_type = diff_src_type>
struct ref_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && this->desc()->diff_dst_desc.data_type == diff_dst_type
                && this->desc()->weights_desc.data_type == wei_type
                && this->desc()->accum_data_type == acc_type
                && this->desc()->diff_src_desc.data_type == diff_src_type
                && is_supported_post_ops();
            return ok ? status::success : status::unimplemented;
        }

        virtual bool support_bias() const override { return true; }

        virtual bool is_supported_post_ops() const {
            const auto &p = this->attr()->post_ops_;
            if (p.len_ > 1)
                return false;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len_; i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::depthwise);
                }
                return ok;
            };

            return all_post_ops_supported();
        }
    };

    ref_convolution_bwd_data_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {
        const auto &post_ops = pd()->attr()->post_ops_;

        for (int i = 0; i < post_ops.len_; i++) {
            auto &post_op = post_ops.entry_[i];
            if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(new ref_depthwise_scalar_fwd_t(post_op.depthwise.alg));
            }
        }
    }

    ~ref_convolution_bwd_data_t() {
        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

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

    nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;
};

template <impl::data_type_t src_type, impl::data_type_t diff_wei_type,
         impl::data_type_t diff_dst_type,
         impl::data_type_t acc_type = diff_wei_type>
struct ref_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_weights
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && this->desc()->src_desc.data_type == src_type
                && this->desc()->diff_weights_desc.data_type == diff_wei_type
                && this->desc()->diff_dst_desc.data_type == diff_dst_type
                && this->desc()->accum_data_type == acc_type
                && IMPLICATION(this->with_bias(),
                        this->desc()->diff_bias_desc.data_type
                        == diff_wei_type)
                && this->attr()->has_default_values();
            return ok ? status::success : status::unimplemented;
        }
    };

    ref_convolution_bwd_weights_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<diff_wei_type>::type diff_wei_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<acc_type>::type acc_data_t;

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
    void execute_backward_weights() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
