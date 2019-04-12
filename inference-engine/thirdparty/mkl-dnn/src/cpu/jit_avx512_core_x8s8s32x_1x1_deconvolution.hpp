
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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_1X1_DECONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_1X1_DECONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_deconvolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "cpu_convolution_pd.hpp"
#include "type_helpers.hpp"
#include "primitive_iterator.hpp"

#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_avx512_core_x8s8s32x_1x1_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t
        : public cpu_primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_) {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_DECONVOLUTION_PD_T(
                jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<src_type,
                        dst_type>);

        status_t init_convolution() {

            convolution_desc_t cd;
            status_t status;

            auto dd = this->desc();
            status = conv_desc_init(&cd, prop_kind::forward_training,
                    alg_kind::convolution_direct, &(dd->src_desc),
                    &(dd->weights_desc), &(dd->bias_desc), &(dd->dst_desc),
                    dd->strides, dd->dilates, dd->padding[0], dd->padding[1],
                    dd->padding_kind);

            if (status == status::success) {
                status = mkldnn_primitive_desc::create<
                        typename mkldnn::impl::cpu::
                                jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<src_type,
                                        dst_type>::pd_t>(&conv_pd_,
                        (op_desc_t *)&cd, &(this->attr_), this->engine_,
                        nullptr);
            }

            if (status == status::success) {
                status = set_default_params();
            }

            return status;
        };

        virtual status_t init() override {
            using namespace prop_kind;
            status_t status;

            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true && utils::one_of(this->desc()->prop_kind,
                                      prop_kind::forward_training,
                                      prop_kind::forward_inference)
                    && this->desc()->alg_kind == alg_kind::deconvolution_direct
                    && !this->has_zero_dim_memory()
                    && this->desc()->src_desc.data_type == src_type
                    && this->desc()->dst_desc.data_type == dst_type
                    && this->desc()->weights_desc.data_type == data_type::s8
                    && IMPLICATION(this->with_bias(),
                               utils::one_of(this->desc()->bias_desc.data_type,
                                           data_type::f32, data_type::s32,
                                           data_type::s8, data_type::u8))
                    && this->desc()->accum_data_type == data_type::s32;

            if (ok)
                status = init_convolution();
            else
                status = status::unimplemented;

            return status;
        }

    protected:
        virtual status_t set_default_params() {
            using namespace memory_format;
            auto conv_1x1_pd_ = static_cast<typename mkldnn::impl::cpu::
                            jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<src_type,
                                    dst_type>::pd_t *>(conv_pd_);
            CHECK(this->src_pd_.set_format(
                    conv_1x1_pd_->src_pd()->desc()->format));
            CHECK(this->dst_pd_.set_format(
                    conv_1x1_pd_->dst_pd()->desc()->format));
            CHECK(this->weights_pd_.set_format(
                    conv_1x1_pd_->weights_pd()->desc()->format));
            if (this->with_bias())
                CHECK(this->bias_pd_.set_format(
                        conv_1x1_pd_->weights_pd(1)->desc()->format));
            return status::success;
        }

        primitive_desc_t *conv_pd_;
        bool conv_supports_bias_;
    };

    jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs), conv_p_(nullptr) {}

    ~jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t() {
        delete this->conv_p_;
    }

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference: (conv_p_)->execute(e); break;
        default: assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
};

}
}
}

#endif /* CPU_JIT_AVX512_CORE_X8S8S32X_1X1_DECONVOLUTION_HPP */
