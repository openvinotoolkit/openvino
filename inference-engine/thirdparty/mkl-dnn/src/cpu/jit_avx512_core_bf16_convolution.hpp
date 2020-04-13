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

#ifndef CPU_JIT_AVX512_CORE_BF16_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_BF16_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_barrier.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_reducer.hpp"

#include "jit_transpose_src_utils.hpp"
#include "jit_avx512_core_bf16_conv_kernel.hpp"
#include "bfloat16_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t dst_type>
struct _jit_avx512_core_bf16_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_()
        {
        }

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_bf16:", avx512_core, ""),
                _jit_avx512_core_bf16_convolution_fwd_t<dst_type>);

        virtual status_t init() override
        {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                    && mayiuse(avx512_core)
                    && utils::one_of(this->desc()->prop_kind, forward_training,
                               forward_inference)
                    && utils::one_of(this->desc()->alg_kind,
                               alg_kind::convolution_auto,
                               alg_kind::convolution_direct)
                    && !this->has_zero_dim_memory()
                    && this->desc()->src_desc.data_type == data_type::bf16
                    && this->desc()->weights_desc.data_type == data_type::bf16
                    && this->desc()->dst_desc.data_type == dst_type
                    && IMPLICATION(this->with_bias(), utils::one_of(
                        this->desc()->bias_desc.data_type, data_type::f32,
                        data_type::bf16));
            if (!ok)
                return status::unimplemented;

            status_t status = jit_avx512_core_bf16_fwd_kernel::init_conf(
                    jcp_, *this->desc(), this->src_pd_, this->weights_pd_,
                    this->dst_pd_, this->bias_pd_, *this->attr(),
                    mkldnn_get_max_threads());
            if (status != status::success) return status;

            if (status == status::success
                    && this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));

            init_scratchpad();

            return status::success;
        }

        inline int ndims() const { return this->desc()->src_desc.ndims; }

        jit_conv_conf_t jcp_;

        private:
            void init_scratchpad() {
                using namespace memory_tracking::names;
                auto scratchpad = scratchpad_registry().registrar();
                if (jcp_.with_bias && jcp_.oc != jcp_.oc_without_padding)
                    scratchpad.book(key_conv_padded_bias,
                        jcp_.typesize_bia * jcp_.oc);
            }
    };

    _jit_avx512_core_bf16_convolution_fwd_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
    {
        kernel_ = new jit_avx512_core_bf16_fwd_kernel(pd()->jcp_,
                    *pd()->attr());
    }
    ~_jit_avx512_core_bf16_convolution_fwd_t() { delete kernel_;}

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual void execute(event_t *e) const {
        if (pd()->ndims() == 3)
            execute_forward_1d();
        else if (pd()->ndims() == 4)
            execute_forward_2d();
        else if (pd()->ndims() == 5)
            execute_forward_3d();
        else
            assert(false);

        /*TODO: zero pad dst */
        e->set_state(event_t::ready);
    }

private:
    void prepare_padded_bias(const char *&bias) const;
    void execute_forward_1d() const;
    void execute_forward_2d() const;
    void execute_forward_3d() const;
    jit_avx512_core_bf16_fwd_kernel *kernel_;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t dst_type>
using jit_avx512_core_bf16_convolution_fwd_t =
    _jit_avx512_core_bf16_convolution_fwd_t<dst_type>;

template <impl::data_type_t diff_src_type>
struct _jit_avx512_core_bf16_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_bf16:", avx512_core, ""),
                _jit_avx512_core_bf16_convolution_bwd_data_t<diff_src_type>);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(avx512_core)
                && utils::one_of(this->desc()->prop_kind, backward_data)
                && utils::one_of(this->desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && this->desc()->diff_dst_desc.data_type == data_type::bf16
                && this->desc()->weights_desc.data_type == data_type::bf16
                && this->desc()->diff_src_desc.data_type == diff_src_type
                && this->set_default_params() == status::success
                && !this->has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status = jit_avx512_core_bf16_bwd_data_kernel::init_conf(
                    jcp_, *this->desc(), *this->diff_src_pd_.desc(),
                    *this->weights_pd_.desc(), *this->diff_dst_pd_.desc());
            if (status != status::success) return status;

            if (status == status::success
                    && this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));

            return status::success;
        }
        jit_conv_conf_t jcp_;
        
    protected:
        memory_format_t src_format()
        {
            using namespace memory_format;
            return utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
        }

        memory_format_t wei_format()
        {
            using namespace memory_format;
            return  this->with_groups()
                ? utils::pick(ndims() - 3, gOIw8o16i2o,
                              gOIhw8o16i2o, gOIdhw8o16i2o)
                : utils::pick(ndims() - 3, OIw8o16i2o,
                              OIhw8o16i2o, OIdhw8o16i2o);
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;

            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            return status::success;
        }
    };

    _jit_avx512_core_bf16_convolution_bwd_data_t(const pd_t *apd,
            const input_vector &inputs, const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs)
    {
        kernel_ = new jit_avx512_core_bf16_bwd_data_kernel(pd()->jcp_);
    }
    ~_jit_avx512_core_bf16_convolution_bwd_data_t() { delete kernel_; };

    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<data_type::bf16>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

    virtual void execute(event_t *e) const {
        if (pd()->ndims() < 5)
            execute_backward_data();
        else if (pd()->ndims() == 5)
            execute_backward_data_3d();
        else
            assert(!"invalid dimension");

        e->set_state(event_t::ready);
    }

private:
    void execute_backward_data() const;
    void execute_backward_data_3d() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_avx512_core_bf16_bwd_data_kernel *kernel_;
};

template <impl::data_type_t diff_src_type>
using jit_avx512_core_bf16_convolution_bwd_data_t =
    _jit_avx512_core_bf16_convolution_bwd_data_t<diff_src_type>;

template <impl::data_type_t diff_weights_type>
struct _jit_avx512_core_bf16_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public  cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_bf16:", avx512_core, ""),
                _jit_avx512_core_bf16_convolution_bwd_weights_t);

        virtual status_t init() override {
            assert(this->engine()->kind() == engine_kind::cpu);
            bool ok = true
                && mayiuse(avx512_core)
                && this->desc()->prop_kind == prop_kind::backward_weights
                && this->desc()->alg_kind == alg_kind::convolution_direct
                && !this->has_zero_dim_memory()
                && this->desc()->src_desc.data_type == data_type::bf16
                && this->desc()->diff_dst_desc.data_type == data_type::bf16
                && this->desc()->diff_weights_desc.data_type
                    == diff_weights_type
                && IMPLICATION(this->with_bias(), utils::one_of(
                    this->desc()->diff_bias_desc.data_type, data_type::f32,
                        data_type::bf16));
           if (!ok) return status::unimplemented;

            status_t status =
                jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf(jcp_,
                        *this->desc(), this->src_pd_, this->diff_weights_pd_,
                        this->diff_bias_pd_, this->diff_dst_pd_);
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);
            return status;
        }

        jit_conv_conf_t jcp_;
        typename cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;

    private:
        void init_balancers() {
            const size_t max_buffer_size = jcp_.nthr * 3 * 5 * 5 * 16 * 16;
            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(jcp_.nthr,
                            jcp_.oc_block, jcp_.ngroups * jcp_.nb_oc, jcp_.mb,
                            max_buffer_size));
            }
        }
    };

    _jit_avx512_core_bf16_convolution_bwd_weights_t(const pd_t *pd,
            const input_vector &inputs, const output_vector &outputs);

    ~_jit_avx512_core_bf16_convolution_bwd_weights_t() {

        delete kernel_;
#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
        delete trans_kernel_;
        delete trans_dst_kernel_;
#endif
        delete acc_ker_;
        delete reducer_bias_;
    }

    typedef typename prec_traits<data_type::bf16>::type src_data_t;
    typedef typename prec_traits<data_type::bf16>::type diff_dst_data_t;
    typedef typename prec_traits<diff_weights_type>::type diff_weights_data_t;

    virtual void execute(event_t *e) const {
        execute_backward_weights();
        e->set_state(event_t::ready);
    }

private:
    struct thread_info_t;
    void execute_backward_weights() const;
    void prepare_scratchpad_data() const;
    void compute_diff_weights(const thread_info_t *) const;
    void reduce_and_convert_diff_weights(const thread_info_t *) const;
    void compute_diff_bias(const thread_info_t *) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    int nthr_, nthr_mb_, nthr_g_, nthr_oc_b_, nthr_ic_b_;

    jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 *kernel_;

    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
    cpu_reducer_t<data_type::f32> *reducer_bias_;

#ifndef BF16_CONV_BWD_W_JIT_KER_USES_PERMW_TRANSPOSITION
    jit_trans_src_t *trans_kernel_;
    jit_trans_dst_t *trans_dst_kernel_;
#endif
};

template <impl::data_type_t diff_src_type>
using jit_avx512_core_bf16_convolution_bwd_weights_t =
    _jit_avx512_core_bf16_convolution_bwd_weights_t<diff_src_type>;

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
