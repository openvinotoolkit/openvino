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
#include "cpu_engine.hpp"
#include "cpu_rnn_pd.hpp"
#include "cpu_isa_traits.hpp"
#include "scratchpad.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "gemm/os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define elemwise_sig(f)                                                 \
    void f(int dic, int wic, int batch, int n_states, int iter_stride, int n_gates, \
            float *ws_gates_, float *states_t_l_, float *states_t_lm1_, \
            float *states_tm1_l_, float *diff_states_t_l_,              \
            float *diff_states_t_lp1_, float *diff_states_tp1_l_,       \
            const float *bias_, float *ws_grid_, float *ws_cell_)

#define cell_execution_sig(f)                                                 \
    void f(int dic, int slc, int sic, int wic, int batch, int n_gates,        \
            int n_states, int iter_stride, float *states_t_l_, float *diff_states_t_l_, \
            float **w_input_, float **w_state_, const float *bias_,           \
            float *states_t_lm1_, float *states_tm1_l_,                       \
            float *diff_states_t_lp1_, float *diff_states_tp1_l_,             \
            float *diff_w_input_, float *diff_w_state_, float *diff_bias_,    \
            float *ws_gates_, float *ws_grid_, float *ws_cell_)

#define grid_execution_sig(f)                                              \
    void f(int dic, int slc, int sic, int wic, int batch, int n_layer,     \
            int n_direction, int n_iter, int n_gates, int n_states,        \
            int n_bias, float **weights_input_, int n_parts_wei_i,         \
            float **weights_states_, int n_parts_wei_st,                   \
            const float *bias_, float *ws_states_, float *ws_diff_states_, \
            float *ws_gates_, float *ws_cell_, float *ws_grid_,            \
            int ws_per_cell, float *diff_weights_layer_,                   \
            float *diff_weights_iter_, float *diff_bias_)

#define gemm_sig(f)                                                          \
    void f(int m, int n, int k, int strideA_m, int strideA_k, int strideB_n, \
            int strideB_k, int strideC_m, int strideC_n, const float *a_,    \
            float *b_, float *c_, bool is_B_trans, float beta)

#define packing_sig(f)                                               \
    void f(int n_layer, int n_direction, int n_weights, int n_gates, \
            int batch, int OC_size, int IC_size, float **weights_,   \
            int n_parts, int *gates_per_part, const float *w_,       \
            float * scratch_mem, bool do_copy)

#define free_packed_sig(f) void f(int n_layer, int n_direction, int n_parts, \
            float **weights_)

template <alg_kind_t alg_kind, prop_kind_t prop_kind>
float activation(float s, float alpha, float cliping, float dd);

template <prop_kind_t aprop>
struct _ref_rnn_common_t : public cpu_primitive_t {
    using class_name = _ref_rnn_common_t<aprop>;
    typedef enum execution_direction_ {
        b2t_l2r,
        b2t_r2l,
        b2t_bi_concat,
        b2t_bi_sum,
        t2b_l2r,
        t2b_r2l,
        t2b_bi_concat,
        t2b_bi_sum
    } execution_direction;
    typedef elemwise_sig((class_name::*elemwise_f));
    typedef cell_execution_sig((class_name::*cell_execution_f));
    typedef grid_execution_sig((class_name::*grid_execution_f));

    typedef gemm_sig((class_name::*gemm_t));
    typedef packing_sig((class_name::*packing_t));
    typedef free_packed_sig((class_name::*free_packed_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        pd_t(engine_t *engine, const rnn_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_pd)
            : base_pd_t(engine, adesc, attr, hint_pd) {}

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init() {
            using namespace prop_kind;
            using namespace utils;
            using namespace memory_format;
            assert(this->engine()->kind() == engine_kind::cpu);
            const alg_kind_t cell_kind = this->desc()->cell_desc.cell_kind;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                               alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                               alg_kind::gru_linear_before_reset)
                    && IMPLICATION(aprop == prop_kind::forward,
                               one_of(this->desc()->prop_kind, forward_training,
                                       forward_inference))
                    && IMPLICATION(aprop == backward,
                               one_of(this->desc()->prop_kind, backward))
                    && this->set_default_params() == status::success;
            if (!ok)
                return status::unimplemented;

            ok = ok && utils::one_of(cell_kind, alg_kind::vanilla_rnn,
                               alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                               alg_kind::gru_linear_before_reset);

            /// @todo check data layouts for all input tensors
            ok = ok && this->desc()->src_layer_desc.format == tnc
                    && this->desc()->dst_layer_desc.format == tnc;

            ok = ok && this->with_bias();
            switch (aprop) {
            case (prop_kind::forward):
                ok = ok && utils::one_of(this->desc()->prop_kind,
                                   forward_training, forward_inference);
                ok = ok && utils::one_of(
                                   this->desc()->weights_layer_desc.format, any,
                                   ldigo, ldigo_p)
                        && utils::one_of(this->desc()->weights_iter_desc.format,
                                   any, ldigo, ldigo_p);
                break;
            case (prop_kind::backward):
                ok = ok && utils::one_of(this->desc()->prop_kind, backward);
                ok = ok && utils::one_of(
                                   this->desc()->weights_layer_desc.format, any,
                                   ldgoi, ldgoi_p)
                        && utils::one_of(this->desc()->weights_iter_desc.format,
                                   any, ldgoi, ldgoi_p);
                break;
            default: ok = false;
            }

            // Check dimensions consistency
            int ls_multiplier
                    = (this->direction() == mkldnn_bidirectional_concat) ? 2 :
                                                                           1;

            ok = ok && (ls_multiplier * this->DIC() == this->DLC())
                    && ((ls_multiplier * this->SLC()) == this->DLC()
                               || (this->L() == 1))
                    && (this->SIC() == this->DIC() || (this->T() == 1));

            // initialize the workspace_pd if needed
            if (this->desc()->prop_kind != forward_inference){
                dims_t ws_dims = { (dim_t)this->get_ws_size() };
                memory_desc_t ws_d;
                mkldnn_memory_desc_init(
                        &ws_d, 1, ws_dims, impl::data_type::f32, memory_format::x);
                this->ws_pd_ = cpu_memory_t::pd_t(this->engine(), &ws_d);
            }

            return ok ? status::success : status::unimplemented;
        }
    };

    _ref_rnn_common_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dic and sic != dic
        /// respectively

        memory_format_t packed_format;
        switch (aprop) {
        case prop_kind::forward_inference:
        case prop_kind::forward_training:
            packed_format = memory_format::ldigo_p;
            break;
        case prop_kind::backward: packed_format = memory_format::ldgoi_p; break;
        default: assert(false);
        }

        merge_gemm_layer = ((aprop == prop_kind::forward) && (conf_.MB() < 128))
            || (aprop == prop_kind::backward);
        merge_gemm_iter = (aprop == prop_kind::backward)
                && (!utils::one_of(conf_.cell_kind(), alg_kind::vanilla_gru,
                            alg_kind::gru_linear_before_reset));
        auto set_pack_funcs = [](bool packed_gemm, gemm_t &g, bool pack_w,
                packing_t &p, free_packed_t &f) {
            g = packed_gemm ? &class_name::packed_gemm : &class_name::gemm;
            p = pack_w ? &class_name::pack_weights :
                             &class_name::no_pack_weights;
            f = pack_w ? &class_name::free_packed_weights :
                             &class_name::free_no_packed_weights;
        };
#ifdef USE_MKL_PACKED_GEMM
        const bool weights_pack_cond =
            (conf_.T() > 1) && (conf_.MB() == 32) &&
            (conf_.SIC() == 512) &&(conf_.SLC() == 512) && (conf_.DIC() == 512);
#else
        const bool weights_pack_cond = false;
#endif

        const bool is_weights_state_packed = conf_.desc()->weights_iter_desc.format == packed_format;
        set_pack_funcs(weights_pack_cond || is_weights_state_packed,
                gemm_state_func, weights_pack_cond && !is_weights_state_packed,
                weights_state_pack_func, weights_state_free_packed_func);

        const bool is_weights_input_packed = conf_.desc()->weights_layer_desc.format == packed_format;
        set_pack_funcs(weights_pack_cond || is_weights_input_packed,
                gemm_input_func, weights_pack_cond && !is_weights_input_packed,
                weights_input_pack_func, weights_input_free_packed_func);

        switch (conf_.cell_kind()) {
        case alg_kind::vanilla_lstm:
            cell_func = &class_name::cell_execution;
            elemwise_func = &class_name::lstm_elemwise;
            break;
        case alg_kind::vanilla_rnn: // @todo switch on cell kind
            cell_func = &class_name::cell_execution;
            elemwise_func = &class_name::rnn_elemwise;
            switch (conf_.activation_kind()) {
            case alg_kind::eltwise_relu:
                activation_func = &activation<alg_kind::eltwise_relu, aprop>;
                break;
            case alg_kind::eltwise_tanh:
                activation_func = &activation<alg_kind::eltwise_tanh, aprop>;
                break;
            case alg_kind::eltwise_logistic:
                activation_func = &activation<alg_kind::eltwise_logistic, aprop>;
                break;
            default: break;
            }
            break;
        case alg_kind::vanilla_gru:
            cell_func = &class_name::cell_execution_gru;
            break;
        case alg_kind::gru_linear_before_reset:
            cell_func = &class_name::cell_execution_gru_lbr;
            elemwise_func = &class_name::gru_lbr_elemwise;
            break;
        default: break;
        }

        n_output_features
                = (conf_.direction() == mkldnn_bidirectional_concat) ? 2 : 1;
        switch (conf_.direction()) {
        case mkldnn_unidirectional_left2right: exec_dir = b2t_l2r; break;
        case mkldnn_unidirectional_right2left: exec_dir = b2t_r2l; break;
        case mkldnn_bidirectional_concat: exec_dir = b2t_bi_concat; break;
        case mkldnn_bidirectional_sum: exec_dir = b2t_bi_sum; break;
        default: break;
        }

        /// @todo put a heuristic to choose between linear execution and
        /// wavefront
        grid_computation = &class_name::linear_execution;

        // we need to allocate memory for:
        // - the states to compute a pass.
        // - the intermediate results from the gates.
        // - the diff_states to compute the backward pass (training only)
        // These should be allocated on scratchpad if fwd inference
        // or on a workspace provided by the user for training.
        /// @todo shall we require the workspace for training or make it
        /// optional?

        // if no worskpace is provided on forward, we use a scratchpad
        // NOTE: here we use a large worskpace for simplicity:
        // - for states:
        //   - TODO: allocate only n_iter * dic + dic for linear execution
        //   (inference)
        //   - TODO: allocate only n_layer_wav * (2*dic) for wavefront
        //   execution (inference)
        // - for gates:
        //   - TODO: allocate only batch * n_gates * dic for linear execution
        //   (inference)
        //   = TODO: allocate only n_layer_wav * batch * n_gates * dic for
        //   wavefront execution (inference)

        use_jit_sgemm_ = ((aprop == prop_kind::forward_inference)
            || (conf_.is_training() && conf_.DIC() < 500))
            && !mayiuse(avx512_mic);

        copy_weights_layer_ = (conf_.WL_LD() != conf_.WL_GLD());
        copy_weights_iter_ = (conf_.WI_LD() != conf_.WI_GLD());

        copy_diff_weights_layer_ = (aprop == prop_kind::backward)
                && (conf_.DWL_LD() != conf_.DWL_GLD());
        copy_diff_weights_iter_ = (aprop == prop_kind::backward)
                && (conf_.DWI_LD() != conf_.DWI_GLD());

        use_workspace_ = (conf_.desc()->prop_kind != prop_kind::forward_inference);

        size_t scratchpad_size = conf_.set_offsets(use_workspace_,
            ws_gates_offset_, ws_states_offset_,  ws_diff_states_offset_,
            ws_grid_comp_offset_,
            conf_.is_lbr(), ws_cell_comp_offset_,
            copy_weights_layer_, ws_weights_layer_offset_,
            copy_weights_iter_, ws_weights_iter_offset_,
            copy_diff_weights_layer_, ws_diff_weights_layer_offset_,
            copy_diff_weights_iter_, ws_diff_weights_iter_offset_);

        scratchpad_ =
            create_scratchpad(scratchpad_size * sizeof(float));

        int max_nparts = (conf_.cell_kind() == alg_kind::vanilla_gru) ? 2 : 1;
        int ptr_wei_sz = conf_.L() * conf_.D() * max_nparts;
        ptr_wei_input_ = (float **)malloc(sizeof(float *) * ptr_wei_sz, 64);
        ptr_wei_state_ = (float **)malloc(sizeof(float *) * ptr_wei_sz, 64);
    }
    ~_ref_rnn_common_t() {
        delete scratchpad_;
        free(ptr_wei_input_);
        free(ptr_wei_state_);
    }

    // typedef typename prec_traits::type data_t;

    virtual void execute(event_t *e) {
        execute_();
        e->set_state(event_t::ready);
    }

private:
    void execute_();
    grid_execution_sig(linear_execution);
    // grid_execution_sig(wavefront_execution);
    cell_execution_sig(cell_execution);
    cell_execution_sig(cell_execution_gru);
    cell_execution_sig(cell_execution_gru_lbr);
    elemwise_sig(rnn_elemwise);
    elemwise_sig(lstm_elemwise);
    elemwise_sig(gru_lbr_elemwise);
    gemm_sig(gemm);
    gemm_sig(packed_gemm);
    packing_sig(pack_weights);
    packing_sig(no_pack_weights);
    free_packed_sig(free_packed_weights);
    free_packed_sig(free_no_packed_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);

    void copy_init_layer(bool lr, bool rl, int n_direction, int n_layer,
            int n_iter, int batch, int slc, int dic, int dlc, int wic,
            int n_states, float *ws_states_, float *ws_diff_states_,
            const float *xt_, const float *diff_dst_layer);
    void copy_init_iter(int n_layer, int n_direction, int n_states, int batch,
            int sic, int dic, int wic, int n_iter, float *ws_states_,
            float *ws_diff_states_, const float *firstit_states_,
            const float *diff_dst_iter);
    void copy_res_layer(bool lr, bool rl, int n_layer, int n_direction,
            int n_iter, int batch, int n_output_features, int slc, int dlc,
            int wic, int n_states, mkldnn_rnn_direction_t direction,
            float *dst_layer_, float *diff_src_layer, const float *ws_states_,
            const float *ws_diff_states_);
    void copy_res_iter(int n_layer, int n_direction, int n_states, int batch,
            int sic, int dic, int wic, int n_iter, float *dst_iter_,
            float *diff_src_iter, const float *ws_states_,
            const float *ws_diff_states_);
    void gates_reduction(int n_gates, int dic, int wic, int batch,
            const float *ws_gates_, float *diff_bias_);
    pd_t conf_;
    bool use_workspace_;
    scratchpad_t *scratchpad_;

    size_t ws_gates_offset_;
    size_t ws_states_offset_;
    size_t ws_weights_layer_offset_;
    size_t ws_weights_iter_offset_;
    size_t ws_diff_states_offset_;
    size_t ws_diff_weights_layer_offset_;
    size_t ws_diff_weights_iter_offset_;
    size_t ws_grid_comp_offset_;
    size_t ws_cell_comp_offset_;

    float *ws_gates_;
    float *ws_states_;
    float *ws_diff_states_;
    float *ws_cell_;
    float *ws_grid_;
    float *ws_weights_layer_;
    float *ws_weights_iter_;
    float *ws_diff_weights_layer_;
    float *ws_diff_weights_iter_;
    int n_output_features;

    float **ptr_wei_input_;
    float **ptr_wei_state_;

    execution_direction exec_dir;
    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    bool copy_weights_layer_;
    bool copy_weights_iter_;
    bool copy_diff_weights_layer_;
    bool copy_diff_weights_iter_;
    bool merge_gemm_layer;
    bool merge_gemm_iter;
    bool use_jit_sgemm_;

    packing_t weights_input_pack_func;
    packing_t weights_state_pack_func;

    gemm_t gemm_input_func;
    gemm_t gemm_state_func;
    elemwise_f elemwise_func;

    free_packed_t weights_input_free_packed_func;
    free_packed_t weights_state_free_packed_func;
};

using ref_rnn_fwd_t = _ref_rnn_common_t<prop_kind::forward>;
using ref_rnn_bwd_t = _ref_rnn_common_t<prop_kind::backward>;
}
}
}
#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
