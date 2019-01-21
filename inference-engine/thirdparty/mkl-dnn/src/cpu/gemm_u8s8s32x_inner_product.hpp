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

#ifndef GEMM_U8S8S32X_INNER_PRODUCT_HPP
#define GEMM_U8S8S32X_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_inner_product_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "scratchpad.hpp"

#include "gemm/os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t dst_type>
struct gemm_u8s8s32x_inner_product_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_fwd_pd_t {
        pd_t(engine_t *engine, const inner_product_desc_t *adesc,
                const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("gemm:blas", gemm_u8s8s32x_inner_product_fwd_t);

        virtual status_t init() override {
            using namespace utils;
            using namespace data_type;

            assert(engine()->kind() == engine_kind::cpu);

            bool ok = true
#if !USE_MKL_IGEMM
                && false
#endif
                && this->set_default_params() == status::success
                && one_of(desc()->prop_kind, prop_kind::forward_training,
                        prop_kind::forward_inference)
                && !has_zero_dim_memory()
                && this->desc()->src_desc.data_type == u8
                && this->desc()->dst_desc.data_type == dst_type
                && this->desc()->weights_desc.data_type == s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8,
                            u8))
                && attr()->post_ops_.len_ <= 1
                && IMPLICATION(attr()->post_ops_.len_,
                        attr()->post_ops_.entry_[0].is_relu(true, false))
                && dense_gemm_consitency_check(src_pd(), weights_pd(),
                        dst_pd());
            return ok ? status::success : status::unimplemented;
        }

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->src_pd_.desc()->format == any)
            {
                if (ndims() == 4) CHECK(this->src_pd_.set_format(nhwc));
                else if (ndims() == 5) CHECK(this->src_pd_.set_format(ndhwc));
                else CHECK(this->src_pd_.set_format(nc));
            }
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nc));
            if (this->weights_pd_.desc()->format == any)
            {
                if (ndims() == 4) CHECK(this->weights_pd_.set_format(hwio));
                else if (ndims() == 5) CHECK(this->weights_pd_.set_format(dhwio));
                else CHECK(this->weights_pd_.set_format(io));
            }
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    gemm_u8s8s32x_inner_product_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), dst_is_acc_(false),
        scratchpad_(nullptr)
    {
        dst_is_acc_ = utils::one_of(dst_type, data_type::s32, data_type::f32);
        if (!dst_is_acc_) {
            size_t size = conf_.MB() * conf_.OC() * sizeof(acc_data_t);
            scratchpad_ = create_scratchpad(size);
        }
    }
    ~gemm_u8s8s32x_inner_product_fwd_t() { delete scratchpad_; };

    typedef typename prec_traits<dst_type>::type data_t;

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    bool dst_is_acc_;
    scratchpad_t *scratchpad_;
};
}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
