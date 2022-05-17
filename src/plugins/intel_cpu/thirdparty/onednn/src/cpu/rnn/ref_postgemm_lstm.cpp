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
 * Cell execution LSTM
 */

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/rnn/postgemm_dispatcher.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace rnn_utils;

template <typename T1, typename T2, typename T3, typename T4,
        typename src_data_t, typename scratch_data_t>
void lstm_fwd_postgemm_template(T1 func1, T2 func2, T3 to_src_dt, T4 to_float,
        const float *scales, const float *cscale,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, src_data_t *ws_gates_,
        scratch_data_t *scratch_gates_, src_data_t *dst_layer_,
        src_data_t *dst_iter_, void *dst_iter_c_, const src_data_t *src_iter_,
        const void *src_iter_c_, const float *weights_peephole_,
        const void *bias_, int block_step) {
    const ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    const scratch_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    const weights_peephole_aoc_t<const float> weights_peephole(
            rnn, weights_peephole_);
    const auto bias_aoc = rnn_utils::make_raw_aoc(
            bias_, types::data_type_size(rnn.bias_dt), rnn.n_bias, rnn.dhc);
    const auto bias = [&](int gate_id, int dhc_id) {
        return rnn_utils::to_float(bias_aoc(gate_id, dhc_id), rnn.bias_dt);
    };
    // If lstmp, instead of dst_layer, we use scratch_ht if inference or ws_ht if training
    const auto dst_layer_ld = rnn.is_lstm_projection
            ? rnn.scratch_ht_ld
            : rnn.dst_layer_ld(cell_position);
    const auto dst_iter_ld = rnn.dst_iter_ld(cell_position);
    const int dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
    const int src_iter_c_ld = rnn.src_iter_c_ld(cell_position);

    const ws_states_layer_aoc<src_data_t> dst_layer(
            rnn, dst_layer_, dst_layer_ld);
    // TODO: we use scratch and not dst_iter for lstmp
    const ws_states_iter_aoc<src_data_t> dst_iter(rnn, dst_iter_, dst_iter_ld);

    const auto dst_iter_c = rnn_utils::make_raw_aoc(dst_iter_c_,
            types::data_type_size(rnn.dst_iter_c_dt), rnn.ws_states_iter_c_nld,
            dst_iter_c_ld);
    const auto src_iter_c_aoc = rnn_utils::make_raw_aoc(src_iter_c_,
            types::data_type_size(rnn.src_iter_c_dt), rnn.ws_states_iter_c_nld,
            src_iter_c_ld);

    const auto src_iter_c = [&](int mb_id, int dhc_id) {
        return rnn_utils::to_float(
                src_iter_c_aoc(mb_id, dhc_id), rnn.src_iter_c_dt);
    };
    const auto dst_iter_c_assign = [&](int mb_id, int dhc_id, float c_state) {
        const auto dst_iter_c_ptr
                = const_cast<void *>(dst_iter_c(mb_id, dhc_id));

        if (rnn.dst_iter_c_dt == data_type::f32)
            *static_cast<float *>(dst_iter_c_ptr) = c_state;
        else if (rnn.dst_iter_c_dt == data_type::bf16)
            *static_cast<bfloat16_t *>(dst_iter_c_ptr)
                    = cpu::saturate_and_round<bfloat16_t>(c_state);
    };

    const auto postgemm_call = [&](int i) {
        const int n_elem = block_step / (int)sizeof(scratch_data_t);
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < n_elem; j++) {
            float gate_i_arg
                    = to_float(scratch_gates(i, 0, j), 0, j) + bias(0, j);
            if (rnn.is_lstm_peephole)
                gate_i_arg += weights_peephole(0, j) * src_iter_c(i, j);

            float gate_f_arg
                    = to_float(scratch_gates(i, 1, j), 1, j) + bias(1, j);
            if (rnn.is_lstm_peephole)
                gate_f_arg += weights_peephole(1, j) * src_iter_c(i, j);

            const float gate_c_arg
                    = to_float(scratch_gates(i, 2, j), 2, j) + bias(2, j);

            // default func1 is sigmoid, func2 is tanh

            const float gate_i = func1(scales + 0, gate_i_arg);
            const float gate_f = func1(scales + 1, gate_f_arg);
            const float gate_c = func2(scales + 2, gate_c_arg);

            const float c_state = gate_f * src_iter_c(i, j) + gate_i * gate_c;
            dst_iter_c_assign(i, j, c_state);

            float gate_o_arg
                    = to_float(scratch_gates(i, 3, j), 3, j) + bias(3, j);
            if (rnn.is_lstm_peephole)
                gate_o_arg += weights_peephole(2, j) * c_state;

            const float gate_o = func1(scales + 3, gate_o_arg);

            const src_data_t ht = to_src_dt(gate_o * func2(cscale, c_state));
            if (dst_layer_ != nullptr) dst_layer(i, j) = ht;
            if (dst_iter_ != nullptr) dst_iter(i, j) = ht;

            // write gates back to memory for training
            // we to_src_dt them as as they are GEMM inputs in BWD
            if (rnn.is_training) {
                ws_gates(i, 0, j) = to_src_dt(gate_i);
                ws_gates(i, 1, j) = to_src_dt(gate_f);
                ws_gates(i, 2, j) = to_src_dt(gate_c);
                ws_gates(i, 3, j) = to_src_dt(gate_o);
            }
        }
    };

    if (rnn.is_brgemm && !rnn.unfused_post_gemm) {
        for (int i = 0; i < rnn.m_block; i++)
            postgemm_call(i);
    } else {
        parallel_nd(rnn.mb, [&](dim_t i) { postgemm_call(i); });
    }
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_f32_t::lstm_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);

    const auto q_id = [&](float f) { return f; };
    const auto deq_id = [&](float f, int i, int j) { return f; };

    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_fwd_postgemm_template(logistic_f, tanh_f, q_id, deq_id, scales,
                cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                dst_layer_, dst_iter_, dst_iter_c_, src_iter_, src_iter_c_,
                weights_peephole_, bias_, block_step);
    else
        lstm_fwd_postgemm_template(linear_f, linear_f, q_id, deq_id, scales,
                cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                dst_layer_, dst_iter_, dst_iter_c_, src_iter_, src_iter_c_,
                weights_peephole_, bias_, block_step);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_bf16_t::lstm_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);
    const auto round_f32_bf16 = [&](float f) { return bfloat16_t(f); };
    const auto deq_id = [&](float f, int i, int j) { return f; };

    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_fwd_postgemm_template(logistic_f, tanh_f, round_f32_bf16, deq_id,
                scales, cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                dst_layer_, dst_iter_, dst_iter_c_, src_iter_, src_iter_c_,
                weights_peephole_, bias_, block_step);
    else
        lstm_fwd_postgemm_template(linear_f, linear_f, round_f32_bf16, deq_id,
                scales, cscale, rnn, cell_position, ws_gates_, scratch_gates_,
                dst_layer_, dst_iter_, dst_iter_c_, src_iter_, src_iter_c_,
                weights_peephole_, bias_, block_step);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_u8_t::lstm_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);

    const float data_shift = pd_->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd_->attr()->rnn_data_qparams_.scale_;

    const auto quantize_f32_u8 = [&](float f) {
        float qf = f * data_scale + data_shift;
        return qz_a1b0<float, dst_layer_t>()(qf);
    };

    const auto dequantize_s32_f32 = [&](gemm_acc_t s, int gate, int j) {
        const float wscale = pd_->attr()->rnn_weights_qparams_.mask_ == 0
                ? weights_scales_[0]
                : weights_scales_[gate * rnn.dhc + j];

        return saturate<float>(s) * (1.f / (wscale * data_scale));
    };

    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_fwd_postgemm_template(logistic_f, tanh_f, quantize_f32_u8,
                dequantize_s32_f32, scales, cscale, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, dst_iter_c_,
                src_iter_, src_iter_c_, weights_peephole_, bias_, block_step);
    else
        lstm_fwd_postgemm_template(linear_f, linear_f, quantize_f32_u8,
                dequantize_s32_f32, scales, cscale, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, dst_iter_c_,
                src_iter_, src_iter_c_, weights_peephole_, bias_, block_step);
}

template <>
rnn_postgemm_sig(rnn_postgemm_fwd_s8_t::lstm_postgemm) {
    const float *scales = pd_->attr()->rnn_tparams_.scales_;
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);

    const float data_shift = pd_->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd_->attr()->rnn_data_qparams_.scale_;

    const auto quantize_f32_s8 = [&](float f) {
        float qf = f * data_scale + data_shift;
        return qz_a1b0<float, dst_layer_t>()(qf);
    };

    const auto dequantize_s32_f32 = [&](gemm_acc_t s, int gate, int j) {
        float wscale = pd_->attr()->rnn_weights_qparams_.mask_ == 0
                ? weights_scales_[0]
                : weights_scales_[gate * rnn.dhc + j];

        return saturate<float>(s) * (1.f / (wscale * data_scale));
    };

    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto logistic_f = [](const float *scale, float a) {
        return logistic_fwd<float>(a);
    };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_fwd_postgemm_template(logistic_f, tanh_f, quantize_f32_s8,
                dequantize_s32_f32, scales, cscale, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, dst_iter_c_,
                src_iter_, src_iter_c_, weights_peephole_, bias_, block_step);
    else
        lstm_fwd_postgemm_template(linear_f, linear_f, quantize_f32_s8,
                dequantize_s32_f32, scales, cscale, rnn, cell_position,
                ws_gates_, scratch_gates_, dst_layer_, dst_iter_, dst_iter_c_,
                src_iter_, src_iter_c_, weights_peephole_, bias_, block_step);
}

template <typename T1, typename T2, typename src_data_t, typename acc_data_t,
        typename scratch_data_t>
void lstm_bwd_postgemm_template(T1 func1, T2 to_src_dt, const float *cscale,
        const rnn_utils::rnn_conf_t &rnn, const cell_position_t cell_position,
        src_data_t *ws_gates_, scratch_data_t *scratch_gates_,
        void *dst_iter_c_, const void *src_iter_c_,
        acc_data_t *diff_src_iter_c_, acc_data_t *diff_dst_layer_,
        acc_data_t *diff_dst_iter_, acc_data_t *diff_dst_iter_c_,
        const float *weights_peephole_, const void *bias_) {
    const ws_gates_aoc<src_data_t> ws_gates(rnn, ws_gates_);
    const ws_gates_aoc<scratch_data_t> scratch_gates(rnn, scratch_gates_);
    const weights_peephole_aoc_t<const float> weights_peephole(
            rnn, weights_peephole_);
    const int dst_iter_c_ld = rnn.dst_iter_c_ld(cell_position);
    const int src_iter_c_ld = rnn.src_iter_c_ld(cell_position);
    const auto src_iter_c_aoc = rnn_utils::make_raw_aoc(src_iter_c_,
            types::data_type_size(rnn.src_iter_c_dt), rnn.ws_states_iter_c_nld,
            src_iter_c_ld);
    const auto src_iter_c = [&](int mb_id, int dhc_id) {
        return rnn_utils::to_float(
                src_iter_c_aoc(mb_id, dhc_id), rnn.src_iter_c_dt);
    };
    const auto dst_iter_c_aoc = rnn_utils::make_raw_aoc(dst_iter_c_,
            types::data_type_size(rnn.dst_iter_c_dt), rnn.ws_states_iter_c_nld,
            dst_iter_c_ld);
    const auto dst_iter_c = [&](int mb_id, int dhc_id) {
        return rnn_utils::to_float(
                dst_iter_c_aoc(mb_id, dhc_id), rnn.dst_iter_c_dt);
    };

    const ws_diff_states_iter_c_aoc<acc_data_t> diff_src_iter_c(
            rnn, diff_src_iter_c_);
    const ws_diff_states_layer_aoc<acc_data_t> diff_dst_layer(
            rnn, diff_dst_layer_);
    const ws_diff_states_iter_aoc<acc_data_t> diff_dst_iter(
            rnn, diff_dst_iter_);
    const ws_diff_states_iter_c_aoc<acc_data_t> diff_dst_iter_c(
            rnn, diff_dst_iter_c_);

    parallel_nd(rnn.mb, [&](dim_t i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < rnn.dhc; j++) {
            const float Ct = dst_iter_c(i, j);
            /// @todo save it in the workspace in fwd pass or recompute it to
            /// save bw
            const float tanhCt = func1(cscale, Ct);
            // we have 2 incoming diffs on Ht if no projection,
            // otherwise we have only 1 as the summation happened
            // before the bwd projection
            float dHt = diff_dst_layer(i, j);
            if (!rnn.is_lstm_projection) dHt += diff_dst_iter(i, j);
            float dCt = diff_dst_iter_c(i, j)
                    + one_m_square(tanhCt) * ws_gates(i, 3, j) * dHt;

            const float dG3 = tanhCt * dHt * x_m_square(ws_gates(i, 3, j));

            if (rnn.is_lstm_peephole) dCt += dG3 * weights_peephole(2, j);

            const float dG1
                    = src_iter_c(i, j) * dCt * x_m_square(ws_gates(i, 1, j));
            const float dG0
                    = ws_gates(i, 2, j) * dCt * x_m_square(ws_gates(i, 0, j));
            const float dG2
                    = ws_gates(i, 0, j) * dCt * one_m_square(ws_gates(i, 2, j));

            diff_src_iter_c(i, j) = dCt * ws_gates(i, 1, j);

            if (rnn.is_lstm_peephole) {
                diff_src_iter_c(i, j) += dG1 * weights_peephole(1, j);
                diff_src_iter_c(i, j) += dG0 * weights_peephole(0, j);
            }

            scratch_gates(i, 0, j) = to_src_dt(dG0);
            scratch_gates(i, 1, j) = to_src_dt(dG1);
            scratch_gates(i, 2, j) = to_src_dt(dG2);
            scratch_gates(i, 3, j) = to_src_dt(dG3);
        }
    });
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_f32_t::lstm_postgemm) {
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);
    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    const auto to_src_dt = [](float a) { return a; };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_bwd_postgemm_template(tanh_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_iter_c_,
                src_iter_c_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                diff_dst_iter_c_, weights_peephole_, bias_);
    else
        lstm_bwd_postgemm_template(linear_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_iter_c_,
                src_iter_c_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                diff_dst_iter_c_, weights_peephole_, bias_);
}

template <>
rnn_postgemm_sig(rnn_postgemm_bwd_bf16_t::lstm_postgemm) {
    const float *cscale = &(pd_->attr()->rnn_tparams_.cscale_);
    const auto linear_f
            = [](const float *scale, float a) { return *scale * a; };
    const auto tanh_f
            = [](const float *scale, float a) { return tanh_fwd<float>(a); };
    const auto to_src_dt = [](float a) { return bfloat16_t(a); };

    if (!pd_->attr()->rnn_tparams_.test_mode_)
        lstm_bwd_postgemm_template(tanh_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_iter_c_,
                src_iter_c_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                diff_dst_iter_c_, weights_peephole_, bias_);
    else
        lstm_bwd_postgemm_template(linear_f, to_src_dt, cscale, rnn,
                cell_position, ws_gates_, scratch_gates_, dst_iter_c_,
                src_iter_c_, diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                diff_dst_iter_c_, weights_peephole_, bias_);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
