/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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
 * Cell execution of Vanilla RNN
 */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/rnn/postgemm_dispatcher.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;

template <>
float activation<alg_kind::eltwise_relu, prop_kind::forward>(
        float s, float alpha, float cliping) {
    return relu_fwd<float>(s, alpha);
}

template <>
float activation<alg_kind::eltwise_relu, prop_kind::backward>(
        float s, float alpha, float cliping) {
    return relu_bwd<float>(s, alpha);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::forward>(
        float s, float alpha, float cliping) {
    return tanh_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::backward>(
        float s, float alpha, float cliping) {
    return one_m_square<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::forward>(
        float s, float alpha, float cliping) {
    return logistic_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::backward>(
        float s, float alpha, float cliping) {
    return x_m_square<float>(s);
}

constexpr float linear(float s, float alpha, float clipping) {
    return alpha * s;
}

template <typename T, typename src_data_t, typename scratch_data_t>
void rnn_fwd_postgemm_template(T func1, const float *scales, float alpha,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *dst_layer_,
        src_data_t *dst_iter_, const src_data_t *src_iter_, const void *bias_,
        int block_step) {

    const ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    const scratch_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    const auto bias_aoc = rnn_utils::make_raw_aoc(
            bias_, types::data_type_size(rnn.bias_dt), rnn.n_bias, rnn.dhc);
    const auto bias = [&](int gate_id, int dhc_id) {
        return to_float(bias_aoc(gate_id, dhc_id), rnn.bias_dt);
    };
    const auto dst_layer_ld = rnn.dst_layer_ld(cell_position);
    const auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    const ws_states_layer_aoc<src_data_t> dst_layer(
            rnn, dst_layer_, dst_layer_ld);
    const ws_states_iter_aoc<src_data_t> dst_iter(rnn, dst_iter_, dst_iter_ld);

    if (scales != nullptr) alpha = scales[0];

    const int n_elem = block_step / sizeof(scratch_data_t);

    const auto postgemm_call = [&](dim_t i) {
        for (int j = 0; j < n_elem; j++) {
            const float h
                    = func1(scratch_gates(i, 0, j) + bias(0, j), alpha, 0);
            if (dst_layer_ != nullptr) dst_layer(i, j) = h;
            if (dst_iter_ != nullptr) dst_iter(i, j) = h;
            if (rnn.is_training) ws_gates(i, 0, j) = h;
        }
    };

    if (rnn.is_brgemm && !rnn.unfused_post_gemm) {
        for (int i = 0; i < rnn.m_block; i++)
            postgemm_call(i);
    } else
        parallel_nd(rnn.mb, postgemm_call);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const auto act_f = [this](float a, float alpha, float clipping) {
        return this->activation_func(a, alpha, clipping);
    };
    const auto linear_f = [](float a, float alpha, float clipping) {
        return linear(a, alpha, clipping);
    };
    const auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_fwd_postgemm_template(act_f, nullptr, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, src_iter_,
                bias_, block_step);
    else
        rnn_fwd_postgemm_template(linear_f, scales, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, src_iter_,
                bias_, block_step);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const auto act_f = [this](float a, float alpha, float clipping) {
        return bfloat16_t(this->activation_func(a, alpha, clipping));
    };
    const auto linear_f = [](float a, float alpha, float clipping) {
        return bfloat16_t(linear(a, alpha, clipping));
    };
    const auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_fwd_postgemm_template(act_f, nullptr, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, src_iter_,
                bias_, block_step);
    else
        rnn_fwd_postgemm_template(linear_f, scales, alpha, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, src_iter_,
                bias_, block_step);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::rnn_postgemm) {
    assert(!"VANILLA RNN int8 is not supported");
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_s8_t::rnn_postgemm) {
    assert(!"VANILLA RNN int8 is not supported");
}

template <typename T1, typename T2, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void rnn_bwd_postgemm_template(T1 func1, T2 to_src, const float *scales,
        float alpha, const rnn_utils::rnn_conf_t &rnn, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, acc_data_t *diff_dst_iter_,
        acc_data_t *diff_dst_layer_) {
    const ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    const ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    const ws_diff_states_iter_aoc<acc_data_t> diff_dst_iter(
            rnn, diff_dst_iter_);
    const ws_diff_states_layer_aoc<acc_data_t> diff_dst_layer(
            rnn, diff_dst_layer_);
    if (scales != nullptr) alpha = scales[0];

    parallel_nd(rnn.mb, [&](dim_t i) {
        for (int j = 0; j < rnn.dhc; ++j) {
            const float dH = diff_dst_layer(i, j) + diff_dst_iter(i, j);
            const auto g = (float)ws_gates(i, 0, j);
            const float res = dH * func1(g, alpha, 0);
            src_data_t res_converted = to_src(res);
            scratch_gates(i, 0, j) = res_converted;
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const auto act_f = [this](float a, float alpha, float clipping) {
        return this->activation_func(a, alpha, 0);
    };
    const auto linear_f = [](float a, float alpha, float clipping) {
        return linear(a, alpha, 0);
    };
    const auto to_src = [&](float a) { return a; };
    const auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_bwd_postgemm_template(act_f, to_src, nullptr, alpha, rnn, ws_gates_,
                scratch_gates_, diff_dst_iter_, diff_dst_layer_);
    else
        rnn_bwd_postgemm_template(linear_f, to_src, scales, alpha, rnn,
                ws_gates_, scratch_gates_, diff_dst_iter_, diff_dst_layer_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::rnn_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const auto act_f = [this](float a, float alpha, float clipping) {
        return this->activation_func(a, alpha, 0);
    };
    const auto linear_f = [](float a, float alpha, float clipping) {
        return linear(a, alpha, 0);
    };
    const auto to_src = [&](float a) { return bfloat16_t(a); };
    const auto alpha = pd_->desc()->alpha;
    if (!pd_->attr()->rnn_tparams_.test_mode_)
        rnn_bwd_postgemm_template(act_f, to_src, nullptr, alpha, rnn, ws_gates_,
                scratch_gates_, diff_dst_iter_, diff_dst_layer_);
    else
        rnn_bwd_postgemm_template(linear_f, to_src, scales, alpha, rnn,
                ws_gates_, scratch_gates_, diff_dst_iter_, diff_dst_layer_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
