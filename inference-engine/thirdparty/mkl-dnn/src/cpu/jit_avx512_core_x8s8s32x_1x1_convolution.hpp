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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_avx512_core_x8s8s32x_1x1_conv_kernel.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template<bool with_relu, impl::data_type_t src_type, impl::data_type_t dst_type>
struct _jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t: public _cpu_convolution_fwd_pd_t<with_relu> {
        pd_t(engine_t *engine,
                const typename pd_t::base_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : _cpu_convolution_fwd_pd_t<with_relu>(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_int8_1x1:", avx512_core, ""),
                _jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<with_relu,
                src_type, dst_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace utils;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->cdesc_().prop_kind, forward_training,
                        forward_inference)
                && this->cdesc_().alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && this->cdesc_().src_desc.data_type == src_type
                && this->cdesc_().dst_desc.data_type == dst_type
                && this->cdesc_().weights_desc.data_type == data_type::s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->cdesc_().bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                && this->cdesc_().accum_data_type == data_type::s32;

            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = &this->cdesc_();
            const memory_desc_t *src_d = this->src_pd_.desc();
            rtus_prepare(this, conv_d, src_d, this->dst_pd_.desc());
            return jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_conf(jcp_,
                    *conv_d, *src_d, *this->weights_pd_.desc(),
                    *this->dst_pd_.desc(), *this->bias_pd_.desc(), *this->attr(),
                    with_relu, this->negative_slope(),
                    mkldnn_get_max_threads(), rtus_.reduce_src_);
        }

        jit_1x1_conv_conf_t jcp_;
        struct reduce_to_unit_stride_t {
            convolution_desc_t conv_d_;
            bool reduce_src_;
        } rtus_;

        protected:
            virtual status_t set_default_params() override {
                using namespace memory_format;
                bool is_sign_input =
                    (this->cdesc_().src_desc.data_type == data_type::s8)
                        ? true : false;
                if (this->src_pd_.desc()->format == any)
                    CHECK(this->src_pd_.set_format(nhwc));
                if (this->dst_pd_.desc()->format == any)
                    CHECK(this->dst_pd_.set_format(nhwc));
                if (this->weights_pd_.desc()->format == any)
                    CHECK(this->weights_pd_.set_format(this->with_groups()
                        ? ((is_sign_input) ? gOIhw4i16o4i_s8s8 : gOIhw4i16o4i)
                        : ((is_sign_input) ? OIhw4i16o4i_s8s8 : OIhw4i16o4i)));
                if (this->bias_pd_.desc()->format == any)
                    CHECK(this->bias_pd_.set_format(x));
                return status::success;
            }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);
    _jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t(const pd_t *pd,
                                          const input_vector &inputs,
                                          const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
        , kernel_(nullptr), rtus_driver_(nullptr), ws_per_thread_(0)
        , scratch_(nullptr), local_scales_(nullptr)
    {
        kernel_ = new jit_avx512_core_x8s8s32x_1x1_conv_kernel(conf_.jcp_,
                    *conf_.attr());
        init_rtus_driver<avx512_common>(this);
        if (conf_.jcp_.signed_input && conf_.jcp_.ver != ver_vnni) {
            size_t scales_size = ((conf_.attr()->output_scales_.count_ == 1)
                    ? 16
                    : conf_.attr()->output_scales_.count_);
            local_scales_ = (float *)malloc(sizeof(float) * scales_size, 64);
            for (size_t i = 0; i < scales_size; i++) {
                local_scales_[i] = conf_.attr()->output_scales_.scales_[i] *
                                        (1.f / conf_.jcp_.wei_adj_scale);
            }
        }
    }
    ~_jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t() {
        delete kernel_;
        delete rtus_driver_;
        free(scratch_);
        if (local_scales_) free(local_scales_);
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

  private:
    void execute_forward();
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const char *bias, dst_data_t *dst);
    pd_t conf_;
    jit_avx512_core_x8s8s32x_1x1_conv_kernel *kernel_;

    rtus_driver_t<avx512_common> *rtus_driver_;
    size_t ws_per_thread_;
    src_data_t *scratch_;
    float* local_scales_;
};

template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t =
    _jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<false, src_type, dst_type>;

template <impl::data_type_t src_type, impl::data_type_t dst_type>
using jit_avx512_core_x8s8s32x_1x1_convolution_relu_t =
    _jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<true, src_type, dst_type>;
}
}
}

#endif
