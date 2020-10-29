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

#ifndef JIT_UNI_RNN_POSTGEMM_DISPATCHER_HPP
#define JIT_UNI_RNN_POSTGEMM_DISPATCHER_HPP

#include "cpu_rnn_pd.hpp"
#include "../cpu_primitive.hpp"
#include "rnn_utils.hpp"

#include "jit_uni_rnn_common_postgemm.hpp"
#include "jit_uni_rnn_cell_postgemm.hpp"
#include "jit_uni_lstm_cell_postgemm.hpp"
#include "jit_uni_gru_lbr_cell_postgemm.hpp"
#include "jit_uni_gru_cell_postgemm_1.hpp"
#include "jit_uni_gru_cell_postgemm_2.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <alg_kind_t alg_kind, prop_kind_t prop_kind>
float activation(float s, float alpha, float cliping, float dd);

template<prop_kind_t aprop, impl::data_type_t src_type>
struct rnn_postgemm_dispatcher {

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename utils::conditional<src_type == data_type::u8, int32_t,
            float>::type acc_data_t;

    using class_name = rnn_postgemm_dispatcher<aprop, src_type>;
    typedef rnn_postgemm_sig((class_name::*postgemm_f));

    rnn_postgemm_dispatcher(const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd): pd_(pd){
        rnn_postgemm_ = nullptr;
        rnn_postgemm_part2_ = nullptr;

        bool jit_path = utils::one_of(pd->desc()->prop_kind,
                prop_kind::forward_inference, prop_kind::forward_training);

        switch (pd->cell_kind()) {
        case alg_kind::vanilla_lstm:
            postgemm_func = &class_name::lstm_postgemm;
            if (jit_path) {
                if (mayiuse(avx512_core))
                    rnn_postgemm_ =
                        new jit_uni_lstm_cell_postgemm_fwd<avx512_core, src_type>(
                            rnn, pd);
                else if (mayiuse(avx2))
                    rnn_postgemm_ =
                        new jit_uni_lstm_cell_postgemm_fwd<avx2, src_type>(
                            rnn, pd);
                else if (mayiuse(sse42))
                    rnn_postgemm_ =
                        new jit_uni_lstm_cell_postgemm_fwd<sse42, src_type>(
                            rnn, pd);
            }
            if (rnn_postgemm_)
                rnn_postgemm_->init();
            else
                postgemm_func = &class_name::lstm_postgemm;
            break;
        case alg_kind::vanilla_rnn:
	    // jitted path
        if (jit_path) {
            if (mayiuse(avx512_core))
                rnn_postgemm_ = new jit_uni_rnn_cell_postgemm_fwd<avx512_core,
                        src_type>(rnn, pd);
            else if (mayiuse(avx2))
                rnn_postgemm_
                        = new jit_uni_rnn_cell_postgemm_fwd<avx2, src_type>(
                                rnn, pd);
            else if (mayiuse(sse42))
                rnn_postgemm_
                        = new jit_uni_rnn_cell_postgemm_fwd<sse42, src_type>(
                                rnn, pd);
        }
            if (rnn_postgemm_)
                rnn_postgemm_->init();
            else {
                // ref path
            postgemm_func = &class_name::rnn_postgemm;
            switch (pd->activation_kind()) {
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
            }
            break;
        case alg_kind::vanilla_gru:
	    // jitted path
        if (jit_path) {
            if (mayiuse(avx512_core)) {
                rnn_postgemm_
                        = new jit_uni_gru_cell_postgemm_part1_fwd<avx512_core,
                                src_type>(rnn, pd);
                rnn_postgemm_part2_
                        = new jit_uni_gru_cell_postgemm_part2_fwd<avx512_core,
                                src_type>(rnn, pd);
            } else if (mayiuse(avx2)) {
                rnn_postgemm_ = new jit_uni_gru_cell_postgemm_part1_fwd<avx2,
                        src_type>(rnn, pd);
                rnn_postgemm_part2_
                        = new jit_uni_gru_cell_postgemm_part2_fwd<avx2,
                                src_type>(rnn, pd);
            } else if (mayiuse(sse42)) {
                rnn_postgemm_ = new jit_uni_gru_cell_postgemm_part1_fwd<sse42,
                        src_type>(rnn, pd);
                rnn_postgemm_part2_
                        = new jit_uni_gru_cell_postgemm_part2_fwd<sse42,
                                src_type>(rnn, pd);
            }
        }
            if (rnn_postgemm_ && rnn_postgemm_part2_) {
                rnn_postgemm_->init();
                rnn_postgemm_part2_->init();
            } else {
                postgemm_func = &class_name::gru_part1_postgemm;
                postgemm_part2_func = &class_name::gru_part2_postgemm;   
            }
            break;
        case alg_kind::gru_linear_before_reset:
            postgemm_func = &class_name::gru_lbr_postgemm;
	    // jitted path
            if (jit_path) {
                if (mayiuse(avx512_core))
                    rnn_postgemm_ =
                        new jit_uni_gru_lbr_cell_postgemm_fwd<avx512_core, src_type>(
                            rnn, pd);
                else if (mayiuse(avx2))
                    rnn_postgemm_ =
                        new jit_uni_gru_lbr_cell_postgemm_fwd<avx2, src_type>(
                            rnn, pd);
                else if (mayiuse(sse42))
                    rnn_postgemm_ =
                        new jit_uni_gru_lbr_cell_postgemm_fwd<sse42, src_type>(
                            rnn, pd);
                assert(rnn_postgemm_ != nullptr);
                rnn_postgemm_->init();
            }
            break;
        default:
            assert(!"Unsupported algorithm kind");
            break;
        }
    }

    ~rnn_postgemm_dispatcher(){
        delete rnn_postgemm_;
        delete rnn_postgemm_part2_;
    }

// template <typename src_data_t, typename acc_data_t>
    rnn_postgemm_sig(execute) {
    if (rnn_postgemm_)
        rnn_postgemm_->execute(rnn, ws_gates_, states_t_l_, c_states_t_l_,
                states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
                diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_,
                ws_cell_);
    else
        (this->*postgemm_func)(rnn, ws_gates_, states_t_l_, c_states_t_l_,
                states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
                diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_,
                ws_cell_);
}

// template <typename src_data_t, typename acc_data_t>
    rnn_postgemm_sig(execute_part2) {
    if(rnn_postgemm_part2_)
        rnn_postgemm_part2_->execute(rnn, ws_gates_, states_t_l_, c_states_t_l_,
                states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
                diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_,
                ws_cell_);
    else
        (this->*postgemm_part2_func)(rnn, ws_gates_, states_t_l_, c_states_t_l_,
                states_tm1_l_, c_states_tm1_l_, diff_states_t_l_,
                diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_,
                ws_cell_);
}


    
private:
    float (*activation_func)(float dd, float s, float alpha, float cliping);
    rnn_postgemm_sig(rnn_postgemm);
    rnn_postgemm_sig(lstm_postgemm);
    rnn_postgemm_sig(gru_part1_postgemm);
    rnn_postgemm_sig(gru_part2_postgemm);
    rnn_postgemm_sig(gru_lbr_postgemm);

    const rnn_pd_t *pd_;
    jit_uni_rnn_postgemm *rnn_postgemm_;
    jit_uni_rnn_postgemm *rnn_postgemm_part2_;
    postgemm_f postgemm_func;
    postgemm_f postgemm_part2_func;
};

using rnn_postgemm_fwd_f32_t = rnn_postgemm_dispatcher<prop_kind::forward, data_type::f32>;
using rnn_postgemm_fwd_u8_t = rnn_postgemm_dispatcher<prop_kind::forward, data_type::u8>;
using rnn_postgemm_bwd_f32_t = rnn_postgemm_dispatcher<prop_kind::backward, data_type::f32>;

}
}
}

#endif
