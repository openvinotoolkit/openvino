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

#ifndef CPU_REF_RNN_HPP
#define CPU_REF_RNN_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "../cpu_isa_traits.hpp"
#include "../gemm/os_blas.hpp"

#include "cpu_rnn_pd.hpp"
#include "rnn_utils.hpp"
#include "jit_uni_rnn_common_postgemm_dispatcher.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <alg_kind_t alg_kind, prop_kind_t prop_kind>
float activation(float s, float alpha, float cliping, float dd);

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type>
struct _ref_rnn_common_t : public cpu_primitive_t {
    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<weights_type>::type weights_data_t;
    typedef typename utils::conditional<src_type == data_type::u8, int32_t,
            float>::type acc_data_t;

    using class_name = _ref_rnn_common_t<aprop, src_type, weights_type>;

    typedef rnn_cell_execution_sig((class_name::*cell_execution_f));
    typedef rnn_grid_execution_sig((class_name::*grid_execution_f));

    typedef rnn_gemm_sig((class_name::*gemm_t));
    typedef rnn_bias_prepare_sig((class_name::*bias_prepare_t));
    typedef rnn_bias_finalize_sig((class_name::*bias_finalize_t));
    typedef rnn_weights_assign_sig((class_name::*weights_assign_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        pd_t(engine_t *engine, const rnn_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::hint_class *hint_pd)
            : base_pd_t(engine, adesc, attr, hint_pd) {}

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init() {
            using namespace prop_kind;
            using namespace utils;
            using namespace memory_format;
            using namespace rnn_utils;
            assert(this->engine()->kind() == engine_kind::cpu);
            const alg_kind_t cell_kind = this->desc()->cell_desc.cell_kind;

            data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                               alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                               alg_kind::gru_linear_before_reset)
                    && IMPLICATION(aprop == prop_kind::forward,
                               one_of(this->desc()->prop_kind, forward_training,
                                           forward_inference))
                    && IMPLICATION(aprop == backward,
                               one_of(this->desc()->prop_kind, backward))
                    && src_layer_dt == src_type
                    && everyone_is(
                               weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && this->with_bias();
            if (!ok)
                return status::unimplemented;

            init_conf(rnn_, *this->desc(), this->src_pd(0), this->src_pd(1),
                    this->weights_pd(0), this->weights_pd(1), this->dst_pd(0));

            if (rnn_.dt_conf == all_f32)
                ok = ok && this->attr()->has_default_values();

            // Set weights descriptors to desired format
            memory_desc_t weights_layer_md = *(this->weights_layer_pd_.desc());
            CHECK(set_expected_desc(rnn_, weights_layer_md, false));
            cpu_memory_t::pd_t new_weights_layer_pd(
                    this->engine_, &weights_layer_md);
            if (this->weights_layer_pd_.desc()->format == any) {
                this->weights_layer_pd_ = new_weights_layer_pd;
            } else if (this->weights_layer_pd_.desc()->format == rnn_packed) {
                if (!this->weights_layer_pd_.is_equal(&new_weights_layer_pd))
                    return status::unimplemented;
            }

            memory_desc_t weights_iter_md = *(this->weights_iter_pd_.desc());
            CHECK(set_expected_desc(rnn_, weights_iter_md, true));
            cpu_memory_t::pd_t new_weights_iter_pd(
                    this->engine_, &weights_iter_md);
            if (this->weights_iter_pd_.desc()->format == any) {
                this->weights_iter_pd_ = new_weights_iter_pd;
            } else if (this->weights_iter_pd_.desc()->format == rnn_packed) {
                if (!this->weights_iter_pd_.is_equal(&new_weights_iter_pd))
                    return status::unimplemented;
            }

            CHECK(this->check_layout_consistency());

            set_conf(rnn_, *this->desc(), this->weights_pd(0),
                    this->weights_pd(1), this->diff_weights_pd(0),
                    this->diff_weights_pd(1));

            size_t scratchpad_sz{0}, ws_sz{0};
            get_scratchpad_and_workspace_sizes(rnn_, scratchpad_sz, ws_sz);

            // initialize the workspace_pd if needed
            if (rnn_.is_training) {
                dims_t ws_dims = { (dim_t)ws_sz };
                memory_desc_t ws_d;
                mkldnn_memory_desc_init(&ws_d, 1, ws_dims, data_type::u8, x);
                this->ws_pd_ = cpu_memory_t::pd_t(this->engine(), &ws_d);
            }

            init_scratchpad(scratchpad_sz);

            return status::success;
        }

        rnn_utils::rnn_conf_t rnn_;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, sizeof(float) * scratchpad_sz, 4096);

            int max_nparts = this->cell_kind() == alg_kind::vanilla_gru ? 2 : 1;
            int ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
            scratchpad.book(key_rnn_ptrs_wei_layer,
                    sizeof(float *) * ptr_wei_sz);
            scratchpad.book(key_rnn_ptrs_wei_iter,
                    sizeof(float *) * ptr_wei_sz);
            scratchpad.book(key_rnn_ptrs_bia,
                    sizeof(float *) * ptr_wei_sz);
        }
    };

    _ref_rnn_common_t(const pd_t *apd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true), rnn_postgemm_(nullptr) {
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dic and sic != dic
        /// respectively

        bias_preparation_func = &class_name::bias_prepare;
        bias_finalization_func = &class_name::bias_finalize;

        auto set_gemm_funcs
                = [](bool packed_gemm, gemm_t &g, weights_assign_t &a) {
                      if (packed_gemm) {
                          g = &class_name::packed_gemm;
                          a = &class_name::assign_packed_weights;
                      } else {
                          g = &class_name::gemm;
                          a = &class_name::assign_weights;
                      }
                  };
        set_gemm_funcs(pd()->rnn_.use_iter_packed_gemm, gemm_iter_func,
                weights_iter_assign_func);

        set_gemm_funcs(pd()->rnn_.use_layer_packed_gemm, gemm_layer_func,
                weights_layer_assign_func);

        rnn_postgemm_ = new rnn_postgemm_dispatcher<aprop, src_type>(pd()->rnn_, pd());
        assert(rnn_postgemm_ != nullptr);
        switch (pd()->cell_kind()) {
        case alg_kind::vanilla_rnn:
        case alg_kind::vanilla_lstm:
            cell_func = &class_name::cell_execution;
            break;
        case alg_kind::vanilla_gru:
            cell_func = &class_name::cell_execution_gru;
            break;
        case alg_kind::gru_linear_before_reset:
            cell_func = &class_name::cell_execution_gru_lbr;
            break;
        default: break;
        }

        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_, ws_gates_offset_, ws_states_offset_,
                ws_c_states_offset_, ws_diff_states_offset_,
                ws_grid_comp_offset_, ws_cell_comp_offset_,
                ws_bias_offset_, scratchpad_size, workspace_size);
    }

    ~_ref_rnn_common_t() {
        delete rnn_postgemm_;
    }

    // typedef typename prec_traits::type data_t;

    virtual void execute(event_t *e) const {
        execute_();
        e->set_state(event_t::ready);
    }

private:
    void execute_() const;
    rnn_grid_execution_sig(linear_execution);
    rnn_cell_execution_sig(cell_execution);
    rnn_cell_execution_sig(cell_execution_gru);
    rnn_cell_execution_sig(cell_execution_gru_lbr);
    rnn_gemm_sig(gemm);
    rnn_gemm_sig(packed_gemm);
    rnn_bias_prepare_sig(bias_prepare);
    rnn_bias_finalize_sig(bias_finalize);
    rnn_weights_assign_sig(assign_weights);
    rnn_weights_assign_sig(assign_packed_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);

    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn,
            src_data_t *ws_states_, float *ws_diff_states_,
            const src_data_t *xt_, const float *diff_dst_layer) const;

    template <typename input_data_t>
    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn,
            src_data_t *ws_states_, float *ws_c_states, float *ws_diff_states_,
            const input_data_t *firstit_states_,
            const float *diff_dst_iter) const;

    template <typename dst_data_t>
    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn,
            dst_data_t *dst_layer_, float *diff_src_layer,
            const src_data_t *ws_states_, const float *ws_diff_states_) const;

    template <typename output_data_t>
    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn,
            output_data_t *dst_iter_, float *diff_src_iter,
            const src_data_t *ws_states_, float *ws_c_states,
            const float *ws_diff_states_) const;

    void gates_reduction(const rnn_utils::rnn_conf_t &rnn,
            const acc_data_t *ws_gates_, float *diff_bias_) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    size_t ws_gates_offset_;
    size_t ws_states_offset_;
    size_t ws_c_states_offset_;
    size_t ws_bias_offset_;
    size_t ws_diff_states_offset_;
    size_t ws_grid_comp_offset_;
    size_t ws_cell_comp_offset_;
    rnn_postgemm_dispatcher<aprop,src_type> *rnn_postgemm_;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    bias_prepare_t bias_preparation_func;
    bias_finalize_t bias_finalization_func;
    weights_assign_t weights_layer_assign_func;
    weights_assign_t weights_iter_assign_func;

    gemm_t gemm_layer_func;
    gemm_t gemm_iter_func;
};

using ref_rnn_fwd_f32_t = _ref_rnn_common_t<prop_kind::forward, data_type::f32, data_type::f32>;
using ref_rnn_bwd_f32_t = _ref_rnn_common_t<prop_kind::backward, data_type::f32, data_type::f32>;
using ref_rnn_fwd_u8s8_t = _ref_rnn_common_t<prop_kind::forward, data_type::u8, data_type::s8>;
}
}
}
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
