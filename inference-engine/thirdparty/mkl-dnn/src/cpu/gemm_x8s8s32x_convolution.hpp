/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef GEMM_X8S8S32X_CONVOLUTION_HPP
#define GEMM_X8S8S32X_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_generator.hpp"
#include "gemm_convolution_utils.hpp"

#include "gemm/gemm.hpp"
#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t src_type, data_type_t dst_type>
struct _gemm_x8s8s32x_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_IMPL_STR,
                _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>);

        virtual status_t init() override {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind,
                        prop_kind::forward_training,
                        prop_kind::forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->src_desc.data_type == src_type
                && this->desc()->dst_desc.data_type == dst_type
                && this->desc()->weights_desc.data_type == s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->desc()->accum_data_type == data_type::s32
                && utils::one_of(this->src_pd_.desc()->format, nhwc, ndhwc)
                && this->src_pd_.desc()->format == this->dst_pd_.desc()->format
                && IMPLICATION(this->src_pd_.desc()->format == nhwc,
                        this->weights_pd_.desc()->format == (this->with_groups()
                                ? ((src_type == data_type::s8) ? hwigo_s8s8 : hwigo)
                                : ((src_type == data_type::s8) ? hwio_s8s8 : hwio)))
                && IMPLICATION(this->src_pd_.desc()->format == ndhwc,
                        this->weights_pd_.desc()->format == (this->with_groups()
                                ? ((src_type == data_type::s8) ? dhwigo_s8s8 : dhwigo)
                                : ((src_type == data_type::s8) ? dhwio_s8s8 : dhwio)))
                && this->is_gemm_conv_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *this->desc(), this->src_pd(), this->weights_pd(0),
                    this->dst_pd(), *this->attr(), mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const size_t ndims_sp = this->desc()->src_desc.ndims - 4;
            return (utils::pick(ndims_sp, nhwc, ndhwc));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const size_t ndims_sp = this->desc()->src_desc.ndims - 4;
            return this->with_groups()
                ? (src_type == data_type::s8) ? utils::pick(ndims_sp, hwigo_s8s8, dhwigo_s8s8)
                                              : utils::pick(ndims_sp, hwigo, dhwigo)
                : (src_type == data_type::s8) ? utils::pick(ndims_sp, hwio_s8s8, dhwio_s8s8)
                                              : utils::pick(ndims_sp, hwio, dhwio);
        }

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
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }

        virtual bool is_gemm_conv_format() const {
            const auto &p = this->attr()->post_ops_;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len_; i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                                             primitive_kind::quantization);
                }
                return ok;
            };

            return all_post_ops_supported();
        }
    };

    _gemm_x8s8s32x_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {
        pp_ker_ = new pp_ker_t(this->pd());
    }
    ~_gemm_x8s8s32x_convolution_fwd_t() {
        delete pp_ker_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    void execute_forward() const;
    // XXX: this is throwaway code that will become unnecessary when we have a
    // sufficiently advanced igemm jit generator that supports quantization,
    // relu, and whatnot
    class pp_ker_t : jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(
        _gemm_x8s8s32x_convolution_fwd_t::pp_kernel);
        pp_ker_t(const pd_t *pd);

        void operator()(dst_data_t *dst, acc_data_t *acc,
            const char *bias, const float *scales,
            float nslope, float sum_scale, float signed_scale,
            int g, size_t start, size_t end, const post_ops_t& p,
            float* weights_zp, int32_t* weights_zp_compensation);

        size_t dst_os_stride_;

    private:
        void generate();

        struct ker_args {
            dst_data_t *dst;
            const acc_data_t *acc;
            const char *bias;
            const float *scales;
            float nslope;
            float sum_scale;
            float signed_scale;
            size_t len;
            size_t oc_offset;
        };
        void(*ker_)(const ker_args *args);

        const jit_gemm_conv_conf_t jcp_;
        size_t OC_;
        size_t OS_;
        data_type_t bias_data_type_;
        size_t bias_data_type_size_;
        size_t scale_idx_mult_;
        round_mode_t rmode_;
        bool do_bias_;
        bool do_relu_;
        bool do_sum_;
        mkldnn::impl::data_type_t sum_data_type_;
        bool do_signed_scaling_;
        bool use_fast_post_processing;
        bool with_weights_zp;
        size_t vlen_;

        nstl::vector<ref_eltwise_scalar_fwd_t*> eltwise_injectors;
        nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;
    };


    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src_base, const wei_data_t *wei_base,
            const char *bia_base, dst_data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;

    int nthr_;
    pp_ker_t *pp_ker_;
};

template <data_type_t dst_type>
struct _gemm_u8s8s32x_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t{
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(IGEMM_S8U8S32_IMPL_STR,
                _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>);

        virtual status_t init() override {
            using namespace data_type;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == prop_kind::backward_data
                && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->diff_src_desc.data_type == dst_type
                && this->desc()->diff_dst_desc.data_type == u8
                && this->desc()->weights_desc.data_type == s8
                && IMPLICATION(this->with_bias(), utils::one_of(
                            this->desc()->bias_desc.data_type, f32, s32, s8,
                            u8))
                && this->desc()->accum_data_type == data_type::s32
                && utils::everyone_is(nhwc, this->diff_src_pd_.desc()->format,
                        this->diff_dst_pd_.desc()->format)
                && this->weights_pd_.desc()->format == (this->with_groups()
                        ? hwigo : hwio)
                && attr()->post_ops_.has_default_values();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *this->desc(), this->diff_src_pd(), this->weights_pd(0),
                    this->diff_dst_pd(), *this->attr(), mkldnn_get_max_threads());
        }

        virtual bool support_bias() const override { return true; }

        jit_gemm_conv_conf_t jcp_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(nhwc));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(nhwc));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(
                            this->with_groups() ? hwigo : hwio));
            if (bias_pd_.desc()->format == any)
                CHECK(bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
             return status::success;
        }
    };

    _gemm_u8s8s32x_convolution_bwd_data_t(const pd_t *apd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {}
    ~_gemm_u8s8s32x_convolution_bwd_data_t() {}

    typedef typename prec_traits<data_type::u8>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type diff_src_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual void execute(event_t *e) const {
        execute_backward_data();
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    void execute_backward_data_thr(const int ithr, const int nthr,
            const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
            const char *bia_base, diff_src_data_t *diff_src_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
