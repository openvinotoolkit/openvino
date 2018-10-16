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

#ifndef _INPUT_LSTM_HPP
#define _INPUT_LSTM_HPP

#include "mkldnn_common.hpp"
#include "rnn/rnn.hpp"

namespace rnn {

/* cfgs definition
arrays:
input,
states,
weights_input,
weights_states,
bias,
dst_last_layer,
dst_last_iteration,
dst_diff_input,
dst_diff_states,
dst_diff_weights_input,
dst_diff_weights_states,
dst_diff_bias,
diff_last_layer,
diff_last_iteration,
params: {data_type, min, max, f_min,* f_max, f_base, f_step, f_sparsity, eps}
*/

const int int_max_exact = 1 << 24;
const _dt_conf_t conf_f32 = {
#if 0
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  2,  2, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  2,  2, 0, 1, .25, 1e-5 },
#elif 0
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
#else
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
#endif
};

const dt_conf_t *str2cfg(const char *str) {
#define CASE(cfg)                         \
    if (!strcasecmp(STRINGIFY(cfg), str)) \
    return CONCAT2(conf_, cfg)
    CASE(f32);
#undef CASE
    []() {
        SAFE(FAIL, CRIT);
        return 0;
    }();
    return (const dt_conf_t *)1;
}

const char *cfg2str(const dt_conf_t *cfg) {
#define CASE(_cfg)                   \
    if (cfg == CONCAT2(conf_, _cfg)) \
    return STRINGIFY(_cfg)
    CASE(f32);
#undef CASE
    []() {
        SAFE(FAIL, CRIT);
        return 0;
    }();
    return NULL;
}

// ?? TODO: need auto filling of strides if chunk without padding to avoid
// specifying strides in each test case

static rnn::rnn_desc_t rnns[] = {
/* alg, activation, direction, sic, slc, dic, dlc, batch, n_layer, n_iter,
 * name */
#if 0
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 32, 32, 32, 32, 4 /*ok with 7 batch*/, 4, 3, "exp-fail-1" },
#endif
    { VANILLA_RNN, TANH, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 1, 1, 1,
            "exp0" },
    { VANILLA_RNN, RELU, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 1, 1, 1,
            "exp1.0" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 1, 1, 1,
            "exp1.1" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 2, 1, 1,
            "exp1.2" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 1, 2, 1,
            "exp1.3" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 1, 1, 2,
            "exp1.4" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_right2left, 1, 1, 1, 1, 1, 1, 1,
            "exp2.1" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_right2left, 1, 1, 1, 1, 2, 1, 1,
            "exp2.2" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_right2left, 1, 1, 1, 1, 1, 2, 1,
            "exp2.3" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_right2left, 1, 1, 1, 1, 1, 1, 2,
            "exp2.4" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_right2left, 2, 2, 2, 2, 4, 5, 7,
            "exp4" },
    { VANILLA_LSTM, TANH, mkldnn_bidirectional_sum, 1, 1, 1, 1, 1, 2, 2,
            "exp5.2" },
    { VANILLA_LSTM, TANH, mkldnn_bidirectional_concat, 1, 1, 1, 1, 1, 1, 1,
            "exp6.2" },
    { VANILLA_LSTM, TANH, mkldnn_bidirectional_concat, 512, 512, 512, 512, 128,
            1, 1, "GNMT-enc-bidir" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 1024, 1024, 1024,
            1024, 128, 7, 2, "GNMT-enc-unidir" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 2, 1, 1, 1, 1, 1, 1,
            "GNMT-dec-1" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 2, 1, 1, 1, 2, 1, 1,
            "GNMT-dec-mb2" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 2, 1, 1, 1, 1, 2, 1,
            "GNMT-dec-l2" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 1, 2, 1, 1, 1, 1, 2,
            "GNMT-dec-i2" },
    { VANILLA_LSTM, TANH, mkldnn_unidirectional_left2right, 2048, 1024, 1024,
            1024, 128, 8, 1, "GNMT-dec" },
    { VANILLA_GRU, TANH, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 1, 1, 1,
            "exp-gru-0" },
    { VANILLA_GRU, TANH, mkldnn_unidirectional_right2left, 1, 2, 1, 1, 1, 1, 4,
            "exp-gru-1" },
    { VANILLA_GRU, TANH, mkldnn_bidirectional_concat, 4, 4, 4, 4, 8, 5, 5,
            "exp-gru-2" },
    { VANILLA_GRU, TANH, mkldnn_bidirectional_sum, 512, 512, 512, 512, 128, 2, 2,
            "exp-gru-3" },
    { VANILLA_GRU, TANH, mkldnn_unidirectional_left2right, 512, 1024, 512, 512, 128, 1, 7,
            "exp-gru-4" },
    { GRU_LINEAR_BEFORE_RESET, TANH, mkldnn_unidirectional_left2right, 1, 1, 1, 1, 1, 1, 1,
            "exp-gru-lbr-0" },
    { GRU_LINEAR_BEFORE_RESET, TANH, mkldnn_unidirectional_right2left, 1, 2, 1, 1, 1, 1, 2,
            "exp-gru-lbr-1" },
    { GRU_LINEAR_BEFORE_RESET, TANH, mkldnn_unidirectional_right2left, 2, 1, 1, 1, 1, 2, 1,
            "exp-gru-lbr-2" },
    { GRU_LINEAR_BEFORE_RESET, TANH, mkldnn_bidirectional_concat, 4, 4, 4, 4, 8, 1, 1,
            "exp-gru-lbr-3" },
    { GRU_LINEAR_BEFORE_RESET, TANH, mkldnn_bidirectional_sum, 512, 512, 512, 512, 128, 2, 2,
            "exp-gru-lbr-4" },
    { GRU_LINEAR_BEFORE_RESET, TANH, mkldnn_unidirectional_left2right, 128, 512, 128, 128, 32, 1, 10,
            "exp-gru-lbr-5" },
    { GRU_LINEAR_BEFORE_RESET, TANH, mkldnn_unidirectional_left2right, 512, 128, 128, 128, 32, 10, 1,
            "exp-gru-lbr-6" },
};

} // namespace rnn

#endif
