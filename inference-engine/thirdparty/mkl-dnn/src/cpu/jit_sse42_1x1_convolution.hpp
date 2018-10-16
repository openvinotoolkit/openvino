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

#ifndef CPU_JIT_SSE42_1x1_CONVOLUTION_HPP
#define CPU_JIT_SSE42_1x1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "jit_sse42_1x1_conv_kernel_f32.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <bool with_relu>
struct _jit_sse42_1x1_convolution_fwd_t: public cpu_primitive_t {
    // TODO: (Roma) Code duplication duplication! Remove with templates
    //              (maybe...)!
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine,
                const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_(), jcp_dw() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", sse42, ""),
                _jit_sse42_1x1_convolution_fwd_t<with_relu>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && utils::everyone_is(data_type::f32,
                        this->cdesc_().src_desc.data_type,
                        this->cdesc_().weights_desc.data_type,
                        this->cdesc_().dst_desc.data_type)
                && utils::implication(this->with_bias(),
                        data_type::f32 == this->cdesc_().bias_desc.data_type);
            if (!ok) return status::unimplemented;

            status_t sts_1x1 = jit_sse42_1x1_conv_kernel_f32::init_conf(jcp_,
                    this->cdesc_(),
                    *this->src_pd_.desc(), *this->weights_pd_.desc(),
                    *this->dst_pd_.desc(), *this->attr(), with_relu,
                    this->negative_slope());
            if (sts_1x1 != status::success) return sts_1x1;

            if (jcp_.with_dw_conv) {
                int dw_conv_oh = (jcp_.oh - ((jcp_.dw_conv_ker_h - 1) + 1) + 2) / jcp_.dw_conv_str_h + 1;
                int dw_conv_ow = (jcp_.ow - ((jcp_.dw_conv_ker_w - 1) + 1) + 2) / jcp_.dw_conv_str_w + 1;

                status_t sts_dw = jit_uni_dw_conv_row_f32<sse42>::init_conf(jcp_dw,
                                                                           jcp_.oc, jcp_.oh, jcp_.ow, dw_conv_oh, dw_conv_ow,
                                                                           jcp_.dw_conv_ker_h, jcp_.dw_conv_ker_w,
                                                                           jcp_.dw_conv_str_h, jcp_.dw_conv_str_w,
                                                                           jcp_.dw_conv_eltwise_alg, jcp_.dw_conv_eltwise_alpha,
                                                                           jcp_.dw_conv_eltwise_beta, jcp_.dw_conv_with_sum);
                if (sts_dw != status::success) return sts_dw;
            }

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        jit_conv_conf_t jcp_dw;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(nChw8c));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(nChw8c));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? gOIhw8i8o : OIhw8i8o));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            return status::success;
        }
    };

    _jit_sse42_1x1_convolution_fwd_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd),
        dw_conv_buffer_size_(0), dw_conv_buffer_(nullptr)
    {
        kernel_ = new jit_sse42_1x1_conv_kernel_f32(conf_.jcp_, *conf_.attr());
        if (conf_.jcp_.with_dw_conv) {
            kernel_dw_ = new jit_uni_dw_conv_row_f32<sse42>(conf_.jcp_dw);

            const int nthreads = omp_get_max_threads();
            dw_conv_buffer_size_ = (size_t) conf_.jcp_dw.kh * conf_.jcp_dw.iw * conf_.jcp_dw.ch_block *
                                   (conf_.jcp_.oc / conf_.jcp_.oc_block);
            dw_conv_buffer_ = (data_t *) malloc(dw_conv_buffer_size_ * nthreads * sizeof(data_t), 64);

        }
    }

    ~_jit_sse42_1x1_convolution_fwd_t() {
        delete kernel_;

        if (conf_.jcp_.with_dw_conv) {
            delete kernel_dw_;
            free(dw_conv_buffer_);
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        if (conf_.jcp_.with_dw_conv)
            execute_forward_fusing();
        else
            execute_forward();

        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    void execute_forward_fusing();

    pd_t conf_;
    jit_sse42_1x1_conv_kernel_f32 *kernel_;
    jit_uni_dw_conv_row_f32<sse42> *kernel_dw_;

    /* fuse with dw conv */
    size_t dw_conv_buffer_size_;
    data_t *dw_conv_buffer_;

};

using jit_sse42_1x1_convolution_fwd_t = _jit_sse42_1x1_convolution_fwd_t<false>;
using jit_sse42_1x1_convolution_relu_t = _jit_sse42_1x1_convolution_fwd_t<true>;

}
}
}

#endif
