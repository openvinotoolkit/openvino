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

#ifndef CPU_JIT_AVX512_COMMON_1x1_CONVOLUTION_HPP
#define CPU_JIT_AVX512_COMMON_1x1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"

#include "jit_avx512_common_1x1_conv_kernel.hpp"
#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_transpose_src_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type,
         impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type>
struct jit_avx512_common_1x1_convolution_fwd_t : public cpu_primitive_t {
    // TODO: (Roma) Code duplication duplication! Remove with templates
    //              (maybe...)!
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx512_common, ""),
                jit_avx512_common_1x1_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace utils;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->src_desc.data_type == src_type
                && this->desc()->weights_desc.data_type == wei_type
                && this->desc()->dst_desc.data_type == dst_type
                && IMPLICATION(this->with_bias(),
                    dst_type == this->desc()->bias_desc.data_type);
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *src_d = this->src_pd_.desc();
            rtus_prepare(this, conv_d, src_d, this->dst_pd_.desc());

            status_t status = jit_avx512_common_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *src_d, *this->weights_pd_.desc(),
                    *this->dst_pd_.desc(), *this->attr(),
                    mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_1x1_conv_kernel::init_scratchpad(scratchpad,
                    jcp_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(pick(this->ndims() - 3,
                    nCw16c, nChw16c)));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(pick(this->ndims() - 3,
                    nCw16c, nChw16c)));
            if (this->weights_pd_.desc()->format == any) {
                if (dst_type == data_type::f32 && src_type == data_type::f32
                    && wei_type == data_type::f32)
                        CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? pick(this->ndims() - 3, gOIw16i16o, gOIhw16i16o)
                            : pick(this->ndims() - 3, OIw16i16o, OIhw16i16o)));
                else if (dst_type == data_type::s32
                    && src_type == data_type::s16
                    && wei_type == data_type::s16)
                        CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? pick(this->ndims() - 3, gOIw8i16o2i, gOIhw8i16o2i)
                            : pick(this->ndims() - 3, OIw8i16o2i, OIhw8i16o2i)));
            }
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_common_1x1_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ =
            new jit_avx512_common_1x1_conv_kernel(pd()->jcp_, *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }

    ~jit_avx512_common_1x1_convolution_fwd_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

  private:
    void execute_forward() const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const dst_data_t *bias, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_common_1x1_conv_kernel *kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
};

using jit_avx512_common_1x1_convolution_fwd_f32_t
        = jit_avx512_common_1x1_convolution_fwd_t<data_type::f32>;
using jit_avx512_common_1x1_convolution_fwd_s16s16s32_t
        = jit_avx512_common_1x1_convolution_fwd_t<data_type::s16,
            data_type::s16, data_type::s32>;

template <impl::data_type_t diff_dst_type,
          impl::data_type_t wei_type = diff_dst_type,
          impl::data_type_t diff_src_type = diff_dst_type>
struct jit_avx512_common_1x1_convolution_bwd_data_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx512_common, ""),
                jit_avx512_common_1x1_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && this->desc()->diff_dst_desc.data_type == diff_dst_type
                && this->desc()->weights_desc.data_type == wei_type
                && this->desc()->diff_src_desc.data_type == diff_src_type;
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *diff_src_d = this->diff_src_pd_.desc();
            rtus_prepare(this, conv_d, diff_src_d, this->diff_dst_pd_.desc());

            status_t status = jit_avx512_common_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *diff_src_d, *this->weights_pd_.desc(),
                    *this->diff_dst_pd_.desc(), *this->attr(),
                    mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_1x1_conv_kernel::init_scratchpad(scratchpad,
                    jcp_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(pick(this->ndims() - 3,
                    nCw16c, nChw16c)));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(pick(this->ndims() - 3,
                   nCw16c, nChw16c)));
            if (this->weights_pd_.desc()->format == any) {
                if (diff_dst_type == data_type::f32
                    && diff_src_type == data_type::f32
                    && wei_type == data_type::f32) {
                    CHECK(this->weights_pd_.set_format(this->with_groups()
                        ? pick(this->ndims() - 3, gIOw16o16i, gIOhw16o16i)
                        : pick(this->ndims() - 3, IOw16o16i, IOhw16o16i)));
                }
                else if (diff_dst_type == data_type::s16
                    && diff_src_type == data_type::s32
                    && wei_type == data_type::s16)
                        CHECK(this->weights_pd_.set_format(this->with_groups()
                            ? pick(this->ndims() - 3, gOIw8o16i2o, gOIhw8o16i2o)
                            : pick(this->ndims() - 3, OIw8o16i2o, OIhw8o16i2o)));
            }
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));

            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_common_1x1_convolution_bwd_data_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ = new jit_avx512_common_1x1_conv_kernel(pd()->jcp_,
                    *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }

    ~jit_avx512_common_1x1_convolution_bwd_data_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

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

    jit_avx512_common_1x1_conv_kernel *kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
};

using jit_avx512_common_1x1_convolution_bwd_data_f32_t
        = jit_avx512_common_1x1_convolution_bwd_data_t<data_type::f32>;
using jit_avx512_common_1x1_convolution_bwd_data_s16s16s32_t
        = jit_avx512_common_1x1_convolution_bwd_data_t<data_type::s16,
            data_type::s16, data_type::s32>;

struct jit_avx512_common_1x1_convolution_bwd_weights_t : public cpu_primitive_t
{
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx512_common, ""),
                jit_avx512_common_1x1_convolution_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_weights
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->diff_weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type)
                && IMPLICATION(this->with_bias(),
                        data_type::f32 == desc()->diff_bias_desc.data_type);
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *src_d = this->src_pd_.desc();
            rtus_prepare(this, conv_d, src_d, this->diff_dst_pd_.desc());

            status_t status = jit_avx512_common_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *src_d, *this->diff_weights_pd_.desc(),
                    *this->diff_dst_pd_.desc(), *this->attr(),
                    mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_1x1_conv_kernel::init_scratchpad(scratchpad,
                    jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        reduce_to_unit_stride_t rtus_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(pick(this->ndims() - 3,
                    nCw16c, nChw16c)));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(pick(this->ndims() - 3,
                    nCw16c, nChw16c)));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(this->with_groups()
                    ? pick(this->ndims() - 3, gOIw16i16o, gOIhw16i16o)
                    : pick(this->ndims() - 3, OIw16i16o, OIhw16i16o)));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }

    private:
        void init_balancers() {
            const size_t max_buffer_size = jcp_.nthr * 3 * 5 * 5 * 16 * 16;
            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(jcp_.nthr,
                            jcp_.oc_block, jcp_.ngroups * jcp_.nb_load,
                            jcp_.mb, max_buffer_size));
            }
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_common_1x1_convolution_bwd_weights_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);

    ~jit_avx512_common_1x1_convolution_bwd_weights_t() {
        delete kernel_;
        delete acc_ker_;
        delete reducer_bias_;
        delete rtus_driver_;
        delete trans_kernel_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

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

    jit_avx512_common_1x1_conv_kernel *kernel_;
    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
    cpu_reducer_t<data_type::f32> *reducer_bias_;
    jit_transpose4x16_src *trans_kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
};

}
}
}

#endif
