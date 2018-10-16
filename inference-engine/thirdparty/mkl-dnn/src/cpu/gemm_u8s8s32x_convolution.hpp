/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef GEMM_U8S8S32X_CONVOLUTION_HPP
#define GEMM_U8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"
#include "gemm_convolution_utils.hpp"

#include "os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu, data_type_t dst_type>
struct _gemm_u8s8s32x_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine, const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd), jcp_() {}

        DECLARE_COMMON_PD_T("gemm:blas",
                _gemm_u8s8s32x_convolution_fwd_t<with_relu, dst_type>);

        virtual status_t init() override {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
#if !USE_MKL_IGEMM
                && false
#endif
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind,
                        prop_kind::forward_training,
                        prop_kind::forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && this->cdesc_().src_desc.data_type == u8
                && this->cdesc_().dst_desc.data_type == dst_type
                && this->cdesc_().weights_desc.data_type == s8
                && utils::implication(this->with_bias(), utils::one_of(
                            this->cdesc_().bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->cdesc_().accum_data_type == data_type::s32
                && utils::everyone_is(nhwc, this->src_pd_.desc()->format,
                        this->dst_pd_.desc()->format)
                && this->weights_pd_.desc()->format == (this->with_groups()
                        ? hwigo : hwio)
                && this->is_gemm_conv_format();

            return ok ? status::success : status::unimplemented;
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nhwc));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nhwc));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? hwigo : hwio));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }

        virtual bool is_gemm_conv_format() const {
            using namespace mkldnn::impl::primitive_kind;
            bool ok = true;
            auto const &po = this->attr()->post_ops_;
            switch (po.len_) {
            case 0: break;
            case 1: ok = ok
                    && (po.entry_[0].is_relu() || po.contain(sum, 0));
                break;
            case 2: ok = ok
                    && (po.contain(sum, 0) && po.entry_[1].is_relu());
                break;
            default: ok = false;
            }
            return ok;
        }
    };

    _gemm_u8s8s32x_convolution_fwd_t(const pd_t *pd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd), col_(nullptr)
        , acc_(nullptr)
    {
        jit_gemm_convolution_utils::init_conf(conf_.jcp_,
            *(conf_.cdesc()), conf_.src_pd(), conf_.weights_pd(0),
            conf_.dst_pd(), with_relu, conf_.negative_slope());

        nthr_ = omp_get_max_threads();
        if (!(utils::everyone_is(1, conf_.jcp_.ic, conf_.jcp_.oc)
                    && conf_.jcp_.ngroups != 1)
                && !(conf_.jcp_.os / nthr_ < 64 && conf_.jcp_.mb != 1))
            nthr_ = 1;

        jit_gemm_convolution_utils::prepare_ws_col<src_data_t>(
                this->conf_.jcp_, &this->col_, nthr_);
        jit_gemm_convolution_utils::prepare_ws_acc<acc_data_t>(
                this->conf_.jcp_, &this->acc_, nthr_);
    }

    ~_gemm_u8s8s32x_convolution_fwd_t() {
        free(this->col_);
        free(this->acc_);
    };

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
    src_data_t *col_;
    acc_data_t *acc_;
    int nthr_;
};

}
}
}

#endif
