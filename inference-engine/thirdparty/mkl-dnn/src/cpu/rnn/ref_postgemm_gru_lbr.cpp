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

/*
 * Cell execution LSTM
 */

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "jit_uni_rnn_common_postgemm_dispatcher.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::gru_lbr_postgemm) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    bias_aoc_t bias(rnn, bias_);
    ws_states_aoc_t states_t_l(rnn, states_t_l_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);
    ws_gates_aoc_t ws_gemm_state(rnn, ws_cell_);
    AOC<float, 2> ws_Wh_b(ws_grid_, rnn.mb, rnn.dic);

    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float Wh_b = ws_gemm_state(i, 2, j) + bias(3, j);
            ws_gates(i, 0, j) = logistic_fwd(
                    ws_gates(i, 0, j) + ws_gemm_state(i, 0, j) + bias(0, j));
            ws_gates(i, 1, j) = logistic_fwd(
                    ws_gates(i, 1, j) + ws_gemm_state(i, 1, j) + bias(1, j));
            ws_gates(i, 2, j) = tanh_fwd(
                    ws_gates(i, 2, j) + ws_gates(i, 1, j) * Wh_b + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0, j)
                    + (1.0f - ws_gates(i, 0, j)) * ws_gates(i, 2, j);
            if (rnn.is_training)
                ws_Wh_b(i, j) = Wh_b;
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::gru_lbr_postgemm) {
    assert(!"GRU LBR int8 is not supported");
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::gru_lbr_postgemm) {
    ws_gates_aoc_t ws_gates(rnn, ws_gates_);
    ws_states_aoc_t states_tm1_l(rnn, states_tm1_l_);
    ws_diff_states_aoc_t diff_states_t_l(rnn, diff_states_t_l_);
    ws_diff_states_aoc_t diff_states_tp1_l(rnn, diff_states_tp1_l_);
    ws_diff_states_aoc_t diff_states_t_lp1(rnn, diff_states_t_lp1_);
    ws_gates_aoc_t ws_gates_r(rnn, ws_cell_);
    AOC<float, 2> ws_Wh_b(ws_grid_, rnn.mb, rnn.dic);

    // 1. calculate dG1 dG2 dG3
    // dG0 = (dht - G2) * dht * (1 - G0) * G0
    // dG1 = (W*h + b) * dG2 * (1 - G1) * G1
    // dG2 = (1 - G0) * dht * (1 - G2*G2)
    parallel_nd(rnn.mb, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, i, j)
                    + diff_states_t_lp1(rnn.n_states, i, j);
            float dG0 = (h - ws_gates(i, 2, j)) * dHt
                    * x_m_square(ws_gates(i, 0, j));
            float dG2 = (1.0f - ws_gates(i, 0, j))
                    * one_m_square(ws_gates(i, 2, j)) * dHt;
            float dG1 = ws_Wh_b(i, j) * dG2 * x_m_square(ws_gates(i, 1, j));

            diff_states_t_l(0, i, j) = dHt * ws_gates(i, 0, j);
            ws_gates(i, 2, j) = dG2;
            ws_gates_r(i, 2, j) = dG2 * ws_gates(i, 1, j);
            ws_gates(i, 0, j) = ws_gates_r(i, 0, j) = dG0;
            ws_gates(i, 1, j) = ws_gates_r(i, 1, j) = dG1;
        }
    });
}

}
}
}
