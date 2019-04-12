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

#ifndef CPU_REF_BINARY_CONVOLUTION_HPP
#define CPU_REF_BINARY_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_binary_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct _ref_binary_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_binary_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_binary_convolution_fwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", _ref_binary_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace data_type;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && this->cdesc_().alg_kind == alg_kind::binary_convolution_direct
                && this->cdesc_().src_desc.data_type == bin
                && this->cdesc_().weights_desc.data_type == bin
                && this->cdesc_().accum_data_type == s32
                && utils::one_of(this->cdesc_().dst_desc.data_type, f32, bin)
                && is_supported_post_ops();
            return ok ? status::success : status::unimplemented;
        }

        virtual bool is_supported_post_ops() const {
            bool ok = true;
            auto const &po = this->attr()->post_ops_;

            auto is_eltwise = [&](int idx) { return po.entry_[idx].is_eltwise(); };
            auto is_depthwise = [&](int idx) { return po.entry_[idx].is_depthwise(); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };
            auto is_simple = [&](int idx) { return (is_eltwise(idx) || is_depthwise(idx)); };
            auto is_binarization = [&](int idx) { return po.entry_[idx].is_binarization(); };

            switch (po.len_) {
            case 0: // no post_ops
                break;
            case 1:
                ok = ok && (is_simple(0) || is_sum(0) || is_binarization(0));
                break;
            case 2:
                ok = ok && ((is_sum(0) && is_simple(1)) || (is_simple(0) && is_simple(1)) ||
                            (is_simple(0) && is_binarization(1)));
                break;
            case 3:
                ok = ok && ((is_sum(0) && is_simple(1) && is_simple(2)) ||
                            (is_simple(0) && is_simple(1) && is_binarization(2)));
                break;

            default: ok = false;
            }
            return ok;
        }
    };

    _ref_binary_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
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

    ~_ref_binary_convolution_fwd_t() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    virtual void execute(event_t *e) const {
        switch (pd()->cdesc()->prop_kind) {
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

using ref_binary_convolution_fwd_t = _ref_binary_convolution_fwd_t;

}
}
}

#endif
