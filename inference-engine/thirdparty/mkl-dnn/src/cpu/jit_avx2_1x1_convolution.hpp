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

#ifndef CPU_JIT_AVX2_1x1_CONVOLUTION_HPP
#define CPU_JIT_AVX2_1x1_CONVOLUTION_HPP

#include <common/primitive_attr.hpp>
#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "cpu_reducer.hpp"

#include "jit_avx2_1x1_conv_kernel_f32.hpp"
#include "jit_uni_1x1_conv_utils.hpp"

#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_1x1_convolution_fwd_t: public cpu_primitive_t {
    // TODO: (Roma) Code duplication duplication! Remove with templates
    //              (maybe...)!
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), jcp_dw_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx2, ""),
                jit_avx2_1x1_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && utils::one_of(this->desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::one_of(this->desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->dst_desc.data_type)
                && IMPLICATION(this->with_bias(),
                        data_type::f32 == this->desc()->bias_desc.data_type)
                && !this->attr()->has_asymmetric_quantization();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *src_d = this->src_pd_.desc();
            rtus_prepare(this, conv_d, src_d, this->dst_pd_.desc());

            status_t sts_1x1 = jit_avx2_1x1_conv_kernel_f32::init_conf(jcp_,
                    *conv_d, *src_d, *this->weights_pd_.desc(),
                    *this->dst_pd_.desc(), *this->attr());
            if (sts_1x1 != status::success) return sts_1x1;

            if (jcp_.with_dw_conv) {
                status_t sts_dw = jit_uni_dw_conv_row_f32<avx2>::init_conf(jcp_, jcp_dw_, *this->attr());
                if (sts_dw != status::success) return sts_dw;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_, jcp_dw_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        jit_conv_conf_t jcp_dw_;
        reduce_to_unit_stride_t rtus_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(utils::pick(this->ndims() - 3,
                    nCw8c, nChw8c)));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(utils::pick(this->ndims() - 3,
                    nCw8c, nChw8c)));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                    ? utils::pick(this->ndims() - 3, gOIw8i8o, gOIhw8i8o)
                    : utils::pick(this->ndims() - 3, OIw8i8o, OIhw8i8o)));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ = new jit_avx2_1x1_conv_kernel_f32(pd()->jcp_, pd()->jcp_dw_, *pd()->attr());
        init_rtus_driver<avx2>(this);

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new jit_uni_dw_conv_row_f32<avx2>(pd()->jcp_dw_, *pd()->attr(), pd()->jcp_dw_.ch_block);
        }
    }

    ~jit_avx2_1x1_convolution_fwd_t() {
        delete kernel_;
        delete rtus_driver_;

        if (pd()->jcp_.with_dw_conv) {
            delete kernel_dw_;
        }
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        if (pd()->jcp_.with_dw_conv)
            execute_forward_with_dw_conv();
        else
            execute_forward();

        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    void execute_forward_with_dw_conv() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx2_1x1_conv_kernel_f32 *kernel_;
    jit_uni_dw_conv_row_f32<avx2> *kernel_dw_;
    rtus_driver_t<avx2> *rtus_driver_;
};

struct jit_avx2_1x1_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx2, ""),
                jit_avx2_1x1_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->diff_src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type);
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = this->desc();
            const memory_desc_t *diff_src_d = this->diff_src_pd_.desc();
            rtus_prepare(this, conv_d, diff_src_d, this->diff_dst_pd_.desc());

            status_t status = jit_avx2_1x1_conv_kernel_f32::init_conf(jcp_,
                    *conv_d, *diff_src_d, *this->weights_pd_.desc(),
                    *this->diff_dst_pd_.desc(), *this->attr());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

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
                CHECK(this->diff_src_pd_.set_format(utils::pick(
                    this->ndims() - 3, nCw8c, nChw8c)));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(utils::pick(
                    this->ndims() - 3, nCw8c, nChw8c)));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(this->with_groups()
                    ? utils::pick(this->ndims() - 3, gOIw8o8i, gOIhw8o8i)
                    : utils::pick(this->ndims() - 3, OIw8o8i, OIhw8o8i)));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_bwd_data_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ = new jit_avx2_1x1_conv_kernel_f32(pd()->jcp_, jit_conv_conf_t(), *pd()->attr());
        init_rtus_driver<avx2>(this);
    }

    ~jit_avx2_1x1_convolution_bwd_data_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

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

    jit_avx2_1x1_conv_kernel_f32 *kernel_;
    rtus_driver_t<avx2> *rtus_driver_;
};

struct jit_avx2_1x1_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx2, ""),
                jit_avx2_1x1_convolution_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_weights
                && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
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

            status_t status = jit_avx2_1x1_conv_kernel_f32::init_conf(jcp_,
                    *conv_d, *src_d, *this->diff_weights_pd_.desc(),
                    *this->diff_dst_pd_.desc(), *this->attr());
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_1x1_conv_kernel_f32::init_scratchpad(scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            auto reducer_wei_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_wei);
            reducer_wei_conf_.init_scratchpad(reducer_wei_scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        cpu_reducer_2d_t<data_type::f32>::conf_t reducer_wei_conf_;
        reduce_to_unit_stride_t rtus_;

    protected:
        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(utils::pick(this->ndims() - 3,
                    nCw8c, nChw8c)));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(utils::pick(
                    this->ndims() - 3, nCw8c, nChw8c)));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(this->with_groups()
                    ? utils::pick(this->ndims() - 3, gOIw8i8o, gOIhw8i8o)
                    : utils::pick(this->ndims() - 3, OIw8i8o, OIhw8i8o)));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }

    private:
        void init_balancers() {
            const int ic_block = jcp_.bcast_block;
            const int nb_ic = jcp_.nb_bcast;
            const int nb_ic_blocking = jcp_.nb_bcast_blocking;
            const int bcast_work = utils::div_up(nb_ic, nb_ic_blocking);

            const int oc_block = jcp_.load_block;
            const int nb_oc = jcp_.nb_load;
            const int nb_oc_blocking = jcp_.nb_load_blocking;
            const int load_work = utils::div_up(nb_oc, nb_oc_blocking);

            const int job_size
                = nb_oc_blocking * nb_ic_blocking * ic_block * oc_block;
            const int njobs_x = bcast_work;
            const int njobs_y = jcp_.ngroups * load_work;

            const int max_threads = mkldnn_get_max_threads();
            const size_t max_buffer_size = max_threads * job_size * 8;

            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(max_threads,
                            oc_block, jcp_.ngroups * jcp_.oc / oc_block,
                            jcp_.mb, max_buffer_size));
            }

            reducer_wei_conf_.init(
                    reduce_balancer_t(max_threads, job_size, njobs_y * njobs_x,
                        jcp_.mb * jcp_.nb_reduce, max_buffer_size),
                    job_size / nb_oc_blocking, nb_oc_blocking, ic_block,
                    nb_ic * ic_block * oc_block, nb_oc);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx2_1x1_convolution_bwd_weights_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs);

    ~jit_avx2_1x1_convolution_bwd_weights_t() {
        delete kernel_;
        delete rtus_driver_;
        delete reducer_weights_;
        delete reducer_bias_;
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

    jit_avx2_1x1_conv_kernel_f32 *kernel_;
    cpu_reducer_2d_t<data_type::f32> *reducer_weights_;
    cpu_reducer_t<data_type::f32> *reducer_bias_;
    rtus_driver_t<avx2> *rtus_driver_;
};

}
}
}

#endif
