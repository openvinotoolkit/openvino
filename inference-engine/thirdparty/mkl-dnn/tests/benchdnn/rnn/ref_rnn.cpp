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

#include <stdlib.h>

#include "src/common/mkldnn_thread.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

namespace rnn {

#define min(a, b) ((a < b) ? a : b)
#define max(a, b) ((a > b) ? a : b)
#define xstr(a) str(a)
#define str(a) #a

#define AOC array_offset_calculator

void lstm_activation(int dic, int n_gates, int batch,
        //    float a[batch][n_gates * wc]
        float *a) {
    AOC<float> pa(a, batch, n_gates, dic);
    mkldnn::impl::parallel_nd(batch, [&](int ib) {
        for (int ih = 0; ih < dic; ih++) {
            pa(ib, 0, ih) = logistic(pa(ib, 0, ih));
            pa(ib, 1, ih) = logistic(pa(ib, 1, ih));
            pa(ib, 2, ih) = tanhf(pa(ib, 2, ih));
            pa(ib, 3, ih) = logistic(pa(ib, 3, ih));
            for (int ig = 0; ig < 4; ig++) {
                print(80, "activation 1 a[%d][%d][%d] = %.7f\n", ib, ig, ih,
                        pa(ib, ig, ih));
            }
        }
    });
}

float activation(activation_t f, float x, bool is_fwd = true) {
    float result = 0;
    switch (f) {
    case RELU: result = is_fwd ? relu(x) : drelu(x); break;
    case LOGISTIC: result = is_fwd ? logistic(x) : dlogistic(x); break;
    case TANH: result = is_fwd ? tanhf(x) : dtanhf(x); break;
    default: assert(!"unknown activation");
    }
    return result;
}

void rnn_fwd(activation_t f, int sic, int slc, int dic, int wc, int batch,
        int n_gates, float *dst_iter_h_, float *gates_,
        const float *weights_layer_, const float *weights_iter_h_,
        const float *bias_, const float *src_layer_, const float *src_iter_h_) {
    AOC<float> dst_iter_h(dst_iter_h_, batch, n_gates, wc);
    AOC<const float> bias(bias_, n_gates, dic);
    AOC<float> gates(gates_, batch, n_gates, dic);

    gemm("C", "N", "N", batch, n_gates * dic, slc, 1.0, src_layer_, wc,
            weights_layer_, n_gates * dic, 0.0, gates_, n_gates * dic);
    gemm("C", "N", "N", batch, n_gates * dic, sic, 1.0, src_iter_h_, wc,
            weights_iter_h_, n_gates * dic, 1.0, gates_, n_gates * dic);

    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n_gates; j++)
            for (int k = 0; k < dic; k++) {
                const auto tmp = activation(f, gates(i, j, k) + bias(j, k));
                gates(i, j, k) = tmp;
                dst_iter_h(i, j, k) = tmp;
            }
}

void gru_fwd(int sic, int slc, int dic, int wc, int batch, int n_gates,
        float *dst_iter_h_, float *gates_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *src_layer_, const float *src_iter_h_) {
    AOC<const float> src_iter_h(src_iter_h_, batch, wc);
    AOC<const float> weights_layer(weights_layer_, slc, n_gates, dic);
    AOC<const float> weights_iter_h(weights_iter_h_, sic, n_gates, dic);
    AOC<const float> bias(bias_, n_gates, dic);
    AOC<float> gates(gates_, batch, n_gates, dic);
    AOC<float> h_dst(dst_iter_h_, batch, wc);

    gemm("C", "N", "N", batch, n_gates * dic, slc, 1.0, src_layer_, wc,
            weights_layer_, n_gates * dic, 0.0, gates_, n_gates * dic);
    gemm("C", "N", "N", batch, (n_gates - 1) * dic, sic, 1.0, src_iter_h_,
            wc, weights_iter_h_, n_gates * dic, 1.0, gates_, n_gates * dic);
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n_gates - 1; j++)
            for (int k = 0; k < dic; k++) {
                gates(i, j, k) = logistic(gates(i, j, k) + bias(j, k));
            }

    for (int i = 0; i < batch; i++)
        for (int k = 0; k < dic; k++) {
            h_dst(i, k) = src_iter_h(i, k) * gates(i, 1, k);
        }

    gemm("C", "N", "N", batch, dic, sic, 1.0, dst_iter_h_, wc,
            &(weights_iter_h(0, 2, 0)), n_gates * dic, 1.0, &(gates(0, 2, 0)),
            n_gates * dic);

    for (int i = 0; i < batch; i++)
        for (int k = 0; k < dic; k++) {
            gates(i, 2, k) = tanhf(gates(i, 2, k) + bias(2, k));
        }

    for (int i = 0; i < batch; i++)
        for (int k = 0; k < dic; k++) {
            h_dst(i, k) = gates(i, 0, k) * src_iter_h(i, k) +
                (1 - gates(i, 0, k)) * gates(i, 2, k);
        }
}

void gru_lbr_fwd(int sic, int slc, int dic, int wc, int batch, int n_gates,
        float *dst_iter_h_, float *gates_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *src_layer_, const float *src_iter_h_,
        float *ws_local_) {
    AOC<const float> src_iter_h(src_iter_h_, batch, wc);
    AOC<const float> weights_layer(weights_layer_, slc, n_gates, dic);
    AOC<const float> weights_iter_h(weights_iter_h_, sic, n_gates, dic);
    AOC<const float> bias(bias_, n_gates + 1, dic);
    AOC<float> gates(gates_, batch, n_gates, dic);
    AOC<float> h_dst(dst_iter_h_, batch, wc);
    AOC<float> tmp_ws(ws_local_, batch, n_gates, dic);

    gemm("C", "N", "N", batch, n_gates * dic, slc, 1.0,  src_layer_, wc,
            weights_layer_, n_gates * dic, 0.0, gates_, n_gates * dic);

    gemm("C", "N", "N", batch, n_gates * dic, sic, 1.0, src_iter_h_, wc,
            weights_iter_h_, n_gates * dic, 0.0, ws_local_, n_gates * dic);

    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n_gates - 1; j++)
            for (int k = 0; k < dic; k++) {
                gates(i, j, k) = logistic(gates(i, j, k) + tmp_ws(i, j, k)
                    + bias(j, k));
            }

    for (int i = 0; i < batch; i++)
        for (int k = 0; k < dic; k++) {
            gates(i, 2, k) = tanhf(gates(i, 2, k) + gates(i, 1, k) * (tmp_ws(i, 2, k)
                + bias(3, k)) + bias(2, k));
        }

    for (int i = 0; i < batch; i++)
        for (int k = 0; k < dic; k++) {
            h_dst(i, k) = gates(i, 0, k) * src_iter_h(i, k) +
                (1 - gates(i, 0, k)) * gates(i, 2, k);
        }

}

// w = [weights_layer | weights_iter] : with order f, i , o, \bar(c)
void lstm_fwd(int sic, int slc, int dic, int wc, int batch, int n_gates,
        float *dst_iter_h_, float *c_dst_, float *gates_,
        const float *weights_layer_, const float *weights_iter_h_,
        const float *bias_, const float *src_layer_, const float *src_iter_h_,
        const float *src_iter_c_) {
    AOC<float> h_dst(dst_iter_h_, batch, wc);
    AOC<float> c_dst(c_dst_, batch, wc);
    AOC<const float> bias(bias_, n_gates, dic);
    AOC<const float> src_iter_c(src_iter_c_, batch, wc);
    AOC<float> gates(gates_, batch, n_gates, dic);

    const int ohi = 0;
    const int ohf = 1;
    const int ohc = 2;
    const int oho = 3;

    gemm("C", "N", "N", batch, n_gates * dic, slc, 1.0, src_layer_, wc,
            weights_layer_, n_gates * dic, 0.0, gates_, n_gates * dic);
    gemm("C", "N", "N", batch, n_gates * dic, sic,1.0, src_iter_h_, wc,
            weights_iter_h_, n_gates * dic, 1.0, gates_, n_gates * dic);

    // add bias
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n_gates; j++)
            for (int k = 0; k < dic; k++) {
                gates(i, j, k) += bias(j, k);
            }

    // run the eltwise
    lstm_activation(dic, n_gates, batch, gates_);

    // compute C_t_l and H_t_l
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < dic; j++) {
            float tmp = gates(i, ohf, j) * src_iter_c(i, j)
                    + gates(i, ohi, j) * gates(i, ohc, j);
            c_dst(i, j) = tmp;
            h_dst(i, j) = gates(i, oho, j) * tanhf(tmp);
        }
}

void rnn_cell_fwd(alg_t alg, activation_t f, int sic, int slc, int dic, int wc,
        int batch, int n_gates, float *dst_iter_h, float *dst_iter_c,
        float *gates, const float *weights_layer, const float *weights_iter,
        const float *bias, const float *src_layer, const float *src_iter_h,
        const float *src_iter_c, float *ws_local_) {
    switch (alg) {
    case VANILLA_GRU:
        gru_fwd(sic, slc, dic, wc, batch, n_gates, dst_iter_h, gates,
                weights_layer, weights_iter, bias, src_layer, src_iter_h);
        break;
    case LBR_GRU:
        gru_lbr_fwd(sic, slc, dic, wc, batch, n_gates, dst_iter_h, gates,
                weights_layer, weights_iter, bias, src_layer, src_iter_h,
                ws_local_);
        break;
    case VANILLA_LSTM:
        lstm_fwd(sic, slc, dic, wc, batch, n_gates, dst_iter_h, dst_iter_c,
                gates, weights_layer, weights_iter, bias, src_layer, src_iter_h,
                src_iter_c);
        break;
    case VANILLA_RNN:
        rnn_fwd(f, sic, slc, dic, wc, batch, n_gates, dst_iter_h, gates,
                weights_layer, weights_iter, bias, src_layer, src_iter_h);
        break;
    default: break;
    }
}
void copy(int dimc, int dimr, int ld_src, int ld_dst, const float *src_,
        float *dst_, rnn_action_t action = action_copy) {
    AOC<const float> src(src_, dimc, ld_src);
    AOC<float> dst(dst_, dimc, ld_dst);

    mkldnn::impl::parallel_nd(dimc, [&](int i) {
        for (int j = 0; j < dimr; j++) {
            dst(i, j) = action == action_sum
                    ? dst(i, j) + src(i, j) : src(i, j);
        }
    });
}

/* FIXME: separate copy_init ???
 * fwd: ws_states = n_states
 * bwd: ws_states = n_states + 1
 *
 * lstm example:
 * fwd: ws keeps {h, c} for every cell
 * bwd: wsb keeps {dh, dc, dx} for every cell
 */
void copy_init(alg_t alg, int sic, int slc, int dic, int dlc, int wc, int batch,
        int n_layer, int n_iter, int n_states, float *ws_,
        const float *src_layer_, const float *firstit_states_,
        rnn_iter_direction_t iter_dir, rnn_layer_direction_t lay_dir,
        int dir_val, int n_dir, bool is_bwd = false, bool is_concat = false) {
    AOC<float> ws(
            ws_, n_layer + 2, n_dir, n_iter + 2, n_states + is_bwd, batch, wc);
    auto c_stride = is_bwd ? (is_concat ? 2 * dlc : dlc) : slc;
    AOC<const float> src_layer(src_layer_, n_iter, batch * c_stride);
    AOC<const float> firstit_states(firstit_states_, n_layer, n_dir, n_states,
            batch, is_bwd ? dic : sic);

    int lay_dest = (lay_dir == bottom2top) ? 0 : n_layer + 1;
    int it_dest = (iter_dir == left2right) ? 0 : n_iter + 1;

    if (!is_bwd) {
        for (int it = 0; it < n_iter; it++)
            copy(batch, slc, slc, wc, &src_layer(it, 0),
                    &ws(lay_dest, dir_val, it + 1, H, 0, 0));

        for (int lay = 0; lay < n_layer; lay++) {
            copy(batch, sic, sic, wc, &firstit_states(lay, dir_val, H, 0, 0),
                    &ws(lay + 1, dir_val, it_dest, H, 0, 0));
            if (alg == VANILLA_LSTM) {
                copy(batch, sic, sic, wc,
                        &firstit_states(lay, dir_val, C, 0, 0),
                        &ws(lay + 1, dir_val, it_dest, C, 0, 0));
            }
        }
    } else {
        for (int it = 0; it < n_iter; it++)
            copy(batch, dic, c_stride, wc,
                    &src_layer(it, dir_val * is_concat * dlc),
                    &ws(lay_dest, dir_val, it + 1, n_states, 0, 0));

        for (int lay = 0; lay < n_layer; lay++) {
            copy(batch, dic, dic, wc, &firstit_states(lay, dir_val, H, 0, 0),
                    &ws(lay + 1, dir_val, it_dest, H, 0, 0));
            if (alg == VANILLA_LSTM) {
                copy(batch, dic, dic, wc,
                        &firstit_states(lay, dir_val, C, 0, 0),
                        &ws(lay + 1, dir_val, it_dest, C, 0, 0));
            }
        }
    }
}

void copy_res(alg_t alg, int sic, int slc, int dic, int dlc, int wc, int batch,
        int n_layer, int n_iter, int n_states, float *lastit_states_,
        float *lastlay_states_, const float *ws_,
        mkldnn_rnn_direction_t direction, rnn_iter_direction_t iter_dir,
        rnn_layer_direction_t lay_dir, int dir_val, int n_dir,
        rnn_action_t action, bool is_bwd = false) {
    int lastlay_c = is_bwd ?
            slc :
            (direction == mkldnn_bidirectional_concat) * dlc + dlc;
    int lastiter_c = is_bwd ? sic : dic;
    AOC<float> lastit_states(
            lastit_states_, n_layer, n_dir, n_states, batch, lastiter_c);
    AOC<float> lastlay_states(lastlay_states_, n_iter, batch, lastlay_c);
    AOC<const float> ws(
            ws_, n_layer + 2, n_dir, n_iter + 2, n_states + is_bwd, batch, wc);
    for (int it = 0; it < n_iter; it++) {
        for (int nb = 0; nb < batch; nb++) {
            // copy H to last layer states
            int lay = is_bwd ? 1 : n_layer;
            int state = is_bwd ? n_states : H;
            auto from = &ws(lay, dir_val, it + 1, state, nb, 0);
            auto to = &lastlay_states(
                    it, nb, (action == action_concat) && (!is_bwd) ? dlc : 0);

            copy(1, is_bwd ?  slc : dlc, wc, lastlay_c, from, to, action);
        }
    }

    int it_source = (iter_dir == left2right) ? n_iter : 1;

    for (int lay = 0; lay < n_layer; lay++) {
        if (alg == VANILLA_LSTM) {
            copy(batch, lastiter_c, wc, lastiter_c,
                    &ws(lay + 1, dir_val, it_source, C, 0, 0),
                    &lastit_states(lay, dir_val, C, 0, 0));
        }
        copy(batch, lastiter_c, wc, lastiter_c,
                &ws(lay + 1, dir_val, it_source, H, 0, 0),
                &lastit_states(lay, dir_val, H, 0, 0));
    }
}

void rnn_linear_fwd(const rnn_prb_t *p, mkldnn_rnn_direction_t direction,
        const float *src_iter_, const float *src_layer_,
        const float *weights_layer_, const float *weights_iter_h_,
        const float *bias_, float *dst_iter_, float *dst_layer_, float *ws_,
        float *gates_) {

    const alg_t alg = p->alg;
    const int sic = p->sic;
    const int slc = p->slc;
    const int dic = p->dic;
    const int dlc = p->dlc;
    const int wc = max(sic, max(slc, dic));
    bool is_lbr = p->alg == LBR_GRU;

    const int batch = p->mb;
    const int n_gates = p->n_gates();
    const int n_states = p->n_states();
    const int n_layer = p->n_layer;
    const int n_iter = p->n_iter;
    const int n_dir = p->n_directions();
    activation_t f = p->activation;

    AOC<const float> bias(bias_, n_layer, n_dir, (n_gates + is_lbr) * dic);
    AOC<const float> weights_layer(
            weights_layer_, n_layer, n_dir, n_gates * dic, slc);
    AOC<const float> weights_iter(
            weights_iter_h_, n_layer, n_dir, n_gates * dic, sic);
    AOC<float> ws(ws_, n_layer + 2, n_dir, n_iter + 2, n_states, batch, wc);
    AOC<float> gates(gates_, n_layer, n_dir, n_iter, batch, n_gates, dic);

    int ws_local_size = is_lbr * batch * n_gates * dic;
    float *ws_local_ = new float[ws_local_size];

    auto process_direction = [&](rnn_iter_direction_t iter_dir,
            rnn_layer_direction_t lay_dir, int dir_val, rnn_action_t action) {
        // we first need to copy the initial states and input into ws
        // it simplifies the logic in the following code
        print(80, "rnn_linear_fwd: call copy_init dir_val = %d\n", dir_val);
        copy_init(alg, sic, slc, dic, dlc, wc, batch, n_layer, n_iter, n_states, ws_,
                src_layer_, src_iter_, iter_dir, lay_dir, dir_val, n_dir);

        // We run the grid of computation
        for (int il = 0; il < n_layer; il++) {
            for (int it = 0; it < n_iter; it++) {
                print(80, "==== layer = %d iter = %d ===\n", il, it);
                int iter = (iter_dir == left2right) ? it + 1 : n_iter - it;
                int prev_iter = (iter_dir == left2right) ? iter - 1 : iter + 1;
                int lay = il + 1;
                rnn_cell_fwd(alg, f, sic, slc, dic, wc, batch, n_gates,
                        &ws(lay, dir_val, iter, H, 0, 0),
                        &ws(lay, dir_val, iter, C, 0, 0),
                        &gates(lay - 1, dir_val, iter - 1, 0, 0, 0),
                        &weights_layer(lay - 1, dir_val, 0, 0),
                        &weights_iter(lay - 1, dir_val, 0, 0),
                        &bias(lay - 1, dir_val, 0),
                        &ws(lay - 1, dir_val, iter, H, 0, 0),
                        &ws(lay, dir_val, prev_iter, H, 0, 0),
                        &ws(lay, dir_val, prev_iter, C, 0, 0),
                        ws_local_);
            }
        }

        // Finally we copy the results to the result buffers
        copy_res(alg, sic, slc, dic, dlc, wc, batch, n_layer, n_iter, n_states,
                dst_iter_, dst_layer_, ws_, direction, iter_dir, lay_dir,
                dir_val, n_dir, action);
    };

    switch (direction) {
    case mkldnn_unidirectional_left2right:
        process_direction(left2right, bottom2top, 0, action_copy);
        break;
    case mkldnn_unidirectional_right2left:
        process_direction(right2left, bottom2top, 0, action_copy);
        break;
    case mkldnn_bidirectional_sum:
        process_direction(left2right, bottom2top, 0, action_copy);
        process_direction(right2left, bottom2top, 1, action_sum);
        break;
    case mkldnn_bidirectional_concat:
        process_direction(left2right, bottom2top, 0, action_copy);
        process_direction(right2left, bottom2top, 1, action_concat);
        break;
    default: assert("unknown direction"); break;
    }

    delete[] ws_local_;
}

void compute_ref_fwd(const rnn_prb_t *p, dnn_mem_t &src_layer_m,
        dnn_mem_t &src_iter_m, dnn_mem_t &weights_src_layer_m,
        dnn_mem_t &weights_src_iter_m, dnn_mem_t &bias_m,
        dnn_mem_t &dst_last_layer_m, dnn_mem_t &dst_last_iteration_m,
        mkldnn_rnn_direction_t direction) {

    assert(direction == mkldnn_unidirectional_left2right
            || direction == mkldnn_unidirectional_right2left
            || direction == mkldnn_bidirectional_sum
            || direction == mkldnn_bidirectional_concat);

    const int wc = max(p->sic, max(p->slc, p->dic));
    int ws_size = (p->n_layer + 2) * p->n_directions() * (p->n_iter + 2)
            * p->n_states() * p->mb * wc;
    auto *ws = new float[ws_size];
    int gates_size = p->n_layer * p->n_directions() * p->n_iter * p->mb
            * p->n_gates() * p->dic;
    auto *gates = new float[gates_size];

    rnn_linear_fwd(p, direction, (float *)src_iter_m, (float *)src_layer_m,
            (float *)weights_src_layer_m, (float *)weights_src_iter_m,
            (float *)bias_m, (float *)dst_last_iteration_m,
            (float *)dst_last_layer_m, ws, gates);

    delete[] ws;
    delete[] gates;
}

// =============================================================================
// ================ BACKWARD ===================================================
// =============================================================================
void rnn_bwd(alg_t alg, activation_t f, int sic, int slc, int dic, int wc,
        int batch, int n_gates, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_h_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *dst_iter_h_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_) {
    AOC<const float> diff_dst_layer(diff_dst_layer_, batch, wc);
    AOC<const float> diff_dst_iter_h(diff_dst_iter_h_, batch, wc);
    AOC<const float> gates(gates_, batch, n_gates, dic);
    AOC<float> b_gates(b_gates_, batch, n_gates, dic);

    for (int b = 0; b < batch; ++b)
        for (int h = 0; h < dic; ++h) {
            const float g = gates(b, 0, h);
            const float dd = diff_dst_layer(b, h) + diff_dst_iter_h(b, h);
            b_gates(b, 0, h) = activation(f, g, false) * dd;
        }

    gemm("C", "T", "N", sic, n_gates * dic, batch, 1.0, src_iter_, wc, b_gates_,
            n_gates * dic, 1.0, diff_weights_iter_h_, n_gates * dic);
    gemm("C", "T", "N", slc, n_gates * dic, batch, 1.0, src_layer_, wc, b_gates_,
            n_gates * dic, 1.0, diff_weights_layer_, n_gates * dic);
    for (int b = 0; b < batch; ++b)
        copy(n_gates, dic, dic, dic, &b_gates(b, 0, 0), diff_bias_, action_sum);

    gemm("C", "N", "T", batch, slc, n_gates * dic, 1.0, b_gates_, n_gates * dic,
            weights_layer_, n_gates * dic, 0.0, diff_src_layer_, wc);
    gemm("C", "N", "T", batch, sic, n_gates * dic, 1.0, b_gates_, n_gates * dic,
            weights_iter_h_, n_gates * dic, 0.0, diff_src_iter_, wc);
}

void lstm_bwd(alg_t alg, int sic, int slc, int dic, int wc, int batch,
        int n_gates, float *diff_src_layer_, float *diff_src_iter_h_,
        float *diff_src_iter_c_, float *diff_weights_layer_,
        float *diff_weights_iter_h_, float *diff_bias_, float *b_gates_,
        const float *src_layer_, const float *src_iter_h_,
        const float *src_iter_c_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *dst_iter_h_, const float *dst_iter_c_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_,
        const float *diff_dst_iter_c_) {
    // TODO: check sic and slc as last dimension in arrays and cycles
    // input
    AOC<const float> diff_dst_layer(diff_dst_layer_, batch, wc);
    AOC<const float> diff_dst_iter_c(diff_dst_iter_c_, batch, wc);
    AOC<const float> diff_dst_iter_h(diff_dst_iter_h_, batch, wc);
    AOC<const float> src_iter_c(src_iter_c_, batch, wc);
    AOC<const float> dst_iter_h(dst_iter_h_, batch, wc);
    AOC<const float> dst_iter_c(dst_iter_c_, batch, wc);
    AOC<const float> gates(gates_, batch, n_gates, dic);

    AOC<float> diff_src_iter_c(diff_src_iter_c_, batch, wc);
    AOC<float> b_gates(b_gates_, batch, n_gates, dic);

    const int ohi = 0;
    const int ohf = 1;
    const int ohc = 2;
    const int oho = 3;

    for (int ib = 0; ib < batch; ib++)
        for (int ih = 0; ih < dic; ih++) {
            print(80, "rnn_single_bwd: ib = %d ih = %d\n", ib, ih);
            float ho = gates(ib, oho, ih);
            float hf = gates(ib, ohf, ih);
            float hc = gates(ib, ohc, ih);
            float hi = gates(ib, ohi, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter_h(ib, ih);
            float c = dst_iter_c(ib, ih);
            float dho = tanhf(c) * dh;
            b_gates(ib, oho, ih) = dlogistic(ho) * dho;

            float dc_next = diff_dst_iter_c(ib, ih);
            float dc = ho * dh * dtanhf(c) + dc_next;
            diff_src_iter_c(ib, ih) = hf * dc;

            float c_old = src_iter_c(ib, ih);
            float dhf = c_old * dc;
            b_gates(ib, ohf, ih) = dlogistic(hf) * dhf;

            float dhi = hc * dc;
            b_gates(ib, ohi, ih) = dlogistic(hi) * dhi;

            float dhc = hi * dc;
            b_gates(ib, ohc, ih) = dtanhf(hc) * dhc;
        }

    gemm("C", "T", "N", sic, n_gates * dic, batch, 1.0, src_iter_h_, wc, b_gates_,
            n_gates * dic, 1.0, diff_weights_iter_h_, n_gates * dic);
    gemm("C", "T", "N", slc, n_gates * dic, batch, 1.0, src_layer_, wc, b_gates_,
            n_gates * dic, 1.0, diff_weights_layer_, n_gates * dic);

    gemm("C", "N", "T", batch, sic, n_gates * dic, 1.0, b_gates_, n_gates * dic,
            weights_iter_h_, n_gates * dic, 0.0, diff_src_iter_h_, wc);
    gemm("C", "N", "T", batch, slc, n_gates * dic, 1.0, b_gates_, n_gates * dic,
            weights_layer_, n_gates * dic, 0.0, diff_src_layer_, wc);

    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n_gates; j++)
            for (int k = 0; k < dic; k++)
                diff_bias_[j * dic + k] += b_gates(i, j, k);
}

void gru_bwd(alg_t alg, activation_t f, int sic, int slc, int dic, int wc,
        int batch, int n_gates, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_h_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *dst_iter_h_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_,
        float *ws_local_) {

    AOC<const float> src_iter(src_iter_, batch, wc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, batch, wc);
    AOC<const float> diff_dst_iter_h(diff_dst_iter_h_, batch, wc);
    AOC<const float> gates(gates_, batch, n_gates, dic);
    AOC<const float> weights_layer(weights_layer_, slc, n_gates, dic);
    AOC<const float> weights_iter_h(weights_iter_h_, sic, n_gates, dic);

    AOC<float> diff_src_iter(diff_src_iter_, batch, wc);
    AOC<float> diff_weights_iter_h(diff_weights_iter_h_, sic, n_gates, dic);
    AOC<float> b_gates(b_gates_, batch, n_gates, dic);

    float *dhr_ = ws_local_;
    float *hr_ = ws_local_ + batch * wc;
    AOC<float> dhr(dhr_, batch, wc);
    AOC<float> hr(hr_, batch, wc);

// dc = (1 - u) * dh; dc^ = dtanhf(c) * dc;
// du = (h - u) * dh; du^ = dlogistic(u) * du;
// dhr = Wc dc^;
// dr = h * dhr; dr^ = dlogistic(r) * dr;
    const int ohu = 0;
    const int ohr = 1;
    const int ohc = 2;
    for (int ib = 0; ib < batch; ib++)
        for (int ih = 0; ih < dic; ih++) {
            float h = src_iter(ib, ih);
            float c = gates(ib, ohc, ih);
            float u = gates(ib, ohu, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter_h(ib, ih);
            float du = (h - c) * dh;
            float dc = (1.0f - u) * dh;
            b_gates(ib, ohu, ih) = dlogistic(u) * du;
            b_gates(ib, ohc, ih) = dtanhf(c) * dc;
            diff_src_iter(ib, ih) = dh * u;
        }
    gemm("C", "N", "T", batch, slc, dic, 1.0, &(b_gates(0, 2, 0)), n_gates * dic,
            &(weights_layer(0, 2, 0)), n_gates * dic, 0.0, dhr_, wc);

    for (int ib = 0; ib < batch; ib++)
        for (int ih = 0; ih < dic; ih++) {
            float h = src_iter(ib, ih);
            float r = gates(ib, ohr, ih);
            float dr = h * dhr(ib, ih);
            hr(ib, ih) = h * r;
            diff_src_iter(ib, ih) += dhr(ib, ih) * r;
            b_gates(ib, ohr, ih) = dlogistic(r) * dr;
        }

// dWx += xdu^ | xdr^ | xdc^
// dWh += hdu^ | ddr^ | (h * r)dc^
    gemm("C", "T", "N", sic, (n_gates - 1) * dic, batch, 1.0, src_iter_, wc,
            b_gates_, n_gates * dic, 1.0, diff_weights_iter_h_, n_gates * dic);
    gemm("C", "T", "N", sic, dic, batch, 1.0, hr_, wc, &(b_gates(0, 2, 0)),
            n_gates * dic, 1.0, &(diff_weights_iter_h(0, 2, 0)), n_gates * dic);
    gemm("C", "T", "N", slc, n_gates * dic, batch, 1.0, src_layer_, wc,
            b_gates_, n_gates * dic, 1.0, diff_weights_layer_, n_gates * dic);

// dx_next = Wxudu^ + Wxrdr^ + Wxcdc^
// dh_next = dh * u + Whudu^ + Whzdz^ + r * Whcdc^
    gemm("C", "N", "T", batch, sic, (n_gates - 1)* dic, 1.0, b_gates_,
            n_gates * dic, weights_iter_h_, n_gates * dic, 1.0, diff_src_iter_,
            wc);
    gemm("C", "N", "T", batch, slc, n_gates * dic, 1.0, b_gates_, n_gates * dic,
            weights_layer_, n_gates * dic, 0.0, diff_src_layer_, wc);

    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n_gates; j++)
            for (int k = 0; k < dic; k++)
                diff_bias_[j * dic + k] += b_gates(i, j, k);
}

void gru_lbr_bwd(alg_t alg, activation_t f, int sic, int slc, int dic, int wc,
        int batch, int n_gates, float *diff_src_layer_, float *diff_src_iter_,
        float *diff_weights_layer_, float *diff_weights_iter_h_,
        float *diff_bias_, float *b_gates_, const float *src_layer_,
        const float *src_iter_, const float *weights_layer_,
        const float *weights_iter_h_, const float *bias_,
        const float *dst_iter_h_, const float *gates_,
        const float *diff_dst_layer_, const float *diff_dst_iter_h_,
        float *ws_local_) {

    AOC<const float> src_iter(src_iter_, batch, wc);
    AOC<const float> diff_dst_layer(diff_dst_layer_, batch, wc);
    AOC<const float> diff_dst_iter_h(diff_dst_iter_h_, batch, wc);
    AOC<const float> gates(gates_, batch, n_gates, dic);
    AOC<const float> weights_layer(weights_layer_, slc, n_gates, dic);
    AOC<const float> weights_iter_h(weights_iter_h_, sic, n_gates, dic);
    AOC<const float> bias(bias_, n_gates + 1, dic);

    AOC<float> diff_src_iter(diff_src_iter_, batch, wc);
    AOC<float> diff_weights_iter_h(diff_weights_iter_h_, dic, n_gates, sic);
    AOC<float> b_gates(b_gates_, batch, n_gates, dic);

    float *Wh_b_ = ws_local_;
    float *b_gates_r_ = ws_local_ + dic * batch;
    AOC<float> Wh_b(Wh_b_, batch, dic);
    AOC<float> b_gates_r(b_gates_r_, batch, n_gates, dic);

    for (int ib = 0; ib < batch; ib++)
        for (int ih = 0; ih < dic; ih++)
            Wh_b(ib, ih) = bias(3, ih);

    gemm("C", "N", "N", batch, dic, sic, 1.0, src_iter_, wc,
            &weights_iter_h(0, 2, 0), n_gates * dic, 1.0, Wh_b_, dic);


// dc = (1 - u) * dh; dc^ = dtanhf(c) * dc;
// du = (h - u) * dh; du^ = dlogistic(u) * du;
// dr = (Wh + b) * dc^; dr^ = dlogistic(r) * dr;
    const int ohu = 0;
    const int ohr = 1;
    const int ohc = 2;
    for (int ib = 0; ib < batch; ib++)
        for (int ih = 0; ih < dic; ih++) {
            float h = src_iter(ib, ih);
            float dh = diff_dst_layer(ib, ih) + diff_dst_iter_h(ib, ih);
            float u = gates(ib, ohu, ih);
            float r = gates(ib, ohr, ih);
            float c = gates(ib, ohc, ih);
            float du = (h - c) * dh;
            float dc = (1.0f - u) * dh;

            b_gates(ib, ohu, ih) = dlogistic(u) * du;
            b_gates(ib, ohc, ih) = dtanhf(c) * dc;

            float dr = Wh_b(ib, ih) * b_gates(ib, ohc, ih);
            b_gates(ib, ohr, ih) = dlogistic(r) * dr;

            b_gates_r(ib, ohu, ih) = b_gates(ib, ohu, ih);
            b_gates_r(ib, ohr, ih) = b_gates(ib, ohr, ih);
            b_gates_r(ib, ohc, ih) = b_gates(ib, ohc, ih) * r;
            diff_src_iter(ib, ih) = dh * u;
        }

    gemm("C", "T", "N", sic, n_gates * dic, batch, 1.0, src_iter_, wc, b_gates_r_,
            n_gates * dic, 1.0, diff_weights_iter_h_, n_gates * dic);
    gemm("C", "T", "N", slc, n_gates * dic, batch, 1.0, src_layer_, wc, b_gates_,
            n_gates * dic, 1.0, diff_weights_layer_, n_gates * dic);

    gemm("C", "N", "T", batch, slc, n_gates * dic, 1.0, b_gates_, n_gates * dic,
            weights_layer_, n_gates * dic, 0.0, diff_src_layer_, wc);
    gemm("C", "N", "T", batch, sic, n_gates * dic, 1.0, b_gates_r_, n_gates * dic,
            weights_iter_h_, n_gates * dic, 1.0, diff_src_iter_, wc);

    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n_gates; j++)
            for (int k = 0; k < dic; k++)
                diff_bias_[j * dic + k] += b_gates(i, j, k);

    for (int i = 0; i < batch; i++)
        for (int k = 0; k < dic; k++)
            diff_bias_[3 * dic + k] += b_gates_r(i, 2, k);
}


void rnn_cell_bwd(alg_t alg, activation_t f, int sic, int slc, int dic, int wc,
        int batch, int n_gates, float *diff_src_layer, float *diff_src_iter_h,
        float *diff_src_iter_c, float *diff_weights_layer,
        float *diff_weights_iter, float *diff_bias, float *b_gates,
        const float *src_layer, const float *src_iter_h,
        const float *src_iter_c, const float *weights_layer,
        const float *weights_iter, const float *bias, const float *dst_iter_h,
        const float *dst_iter_c, const float *gates,
        const float *diff_dst_layer, const float *diff_dst_iter_h,
        const float *diff_dst_iter_c, float *ws_local_) {

    switch (alg) {
    case VANILLA_LSTM:
        lstm_bwd(alg, sic, slc, dic, wc, batch, n_gates, diff_src_layer,
                diff_src_iter_h, diff_src_iter_c, diff_weights_layer,
                diff_weights_iter, diff_bias, b_gates, src_layer, src_iter_h,
                src_iter_c, weights_layer, weights_iter, bias, dst_iter_h,
                dst_iter_c, gates, diff_dst_layer, diff_dst_iter_h,
                diff_dst_iter_c);
        break;
    case VANILLA_RNN:
        rnn_bwd(alg, f, sic, slc, dic, wc, batch, n_gates, diff_src_layer,
                diff_src_iter_h, diff_weights_layer, diff_weights_iter,
                diff_bias, b_gates, src_layer, src_iter_h, weights_layer,
                weights_iter, bias, dst_iter_h, gates, diff_dst_layer,
                diff_dst_iter_h);
        break;
    case VANILLA_GRU:
        gru_bwd(alg, f, sic, slc, dic, wc, batch, n_gates, diff_src_layer,
                diff_src_iter_h, diff_weights_layer, diff_weights_iter,
                diff_bias, b_gates, src_layer, src_iter_h, weights_layer,
                weights_iter, bias, dst_iter_h, gates, diff_dst_layer,
                diff_dst_iter_h, ws_local_);
        break;
    case LBR_GRU:
        gru_lbr_bwd(alg, f, sic, slc, dic, wc, batch, n_gates, diff_src_layer,
                diff_src_iter_h, diff_weights_layer, diff_weights_iter,
                diff_bias, b_gates, src_layer, src_iter_h, weights_layer,
                weights_iter, bias, dst_iter_h, gates, diff_dst_layer,
                diff_dst_iter_h, ws_local_);
    default: break;
    }
}

void rnn_linear_bwd(const rnn_prb_t *p, mkldnn_rnn_direction_t direction,
        const float *diff_dst_iter_, const float *diff_dst_layer_,
        const float *weights_layer_, const float *weights_iter_h_,
        const float *bias_, float *diff_src_iter_, float *diff_src_layer_,
        float *diff_weights_layer_, float *diff_weights_iter_h_,
        float *diff_bias_, float *ws_, const float *gates_) {

    const alg_t alg = p->alg;
    const int sic = p->sic;
    const int slc = p->slc;
    const int dic = p->dic;
    const int dlc = p->dlc;
    const int wc = max(sic, max(slc, dic));
    bool is_lbr = p->alg == LBR_GRU;

    const int batch = p->mb;
    const int n_gates = p->n_gates();
    const int n_states = p->n_states();
    const int n_layer = p->n_layer;
    const int n_iter = p->n_iter;
    const int n_dir = p->n_directions();
    activation_t f = p->activation;

    const int X = n_states;

    AOC<const float> bias(bias_, n_layer, n_dir, n_gates + is_lbr, dic);
    AOC<float> diff_bias(diff_bias_, n_layer, n_dir, n_gates + is_lbr, dic);

    AOC<const float> weights_layer(
            weights_layer_, n_layer, n_dir, n_gates * dic, slc);
    AOC<const float> weights_iter(
            weights_iter_h_, n_layer, n_dir, n_gates * dic, sic);

    AOC<float> diff_weights_layer(
            diff_weights_layer_, n_layer, n_dir, n_gates * dic, slc);
    AOC<float> diff_weights_iter(
            diff_weights_iter_h_, n_layer, n_dir, n_gates * dic, sic);

    auto *b_gates = new float[batch * n_gates * dic];
    AOC<float> ws(ws_, n_layer + 2, n_dir, n_iter + 2, n_states, batch, wc);
    AOC<const float> gates(gates_, n_layer, n_dir, n_iter, batch, n_gates, dic);

    int wsb_size = (n_layer + 2) * n_dir * (n_iter + 2) * (n_states + 1) * batch
            * wc;
    auto *wsb_ = new float[wsb_size];
    init_buffer(wsb_, wsb_size, 0.); // ??!! Temporary. For debug.
    // n_states + 1  -- H, C, X
    AOC<float> wsb(
            wsb_, n_layer + 2, n_dir, n_iter + 2, n_states + 1, batch, wc);

    int ws_local_size;
    switch (p->alg) {
        case LBR_GRU:
            ws_local_size = batch * (n_gates + 1) * dic;
            break;
        case VANILLA_GRU:
            ws_local_size = 2 * batch * wc;
            break;
        default: ws_local_size = 0;
    }
    float *ws_local_ = new float[ws_local_size];

    auto process_direction = [&](rnn_iter_direction_t iter_dir,
            rnn_layer_direction_t lay_dir, int dir_val, rnn_action_t action) {
        // we first need to copy the initial states and input into ws
        // it simplifies the logic in the following code
        copy_init(alg, sic, slc, dic, dlc, wc, batch, n_layer, n_iter, n_states,
                wsb_, diff_dst_layer_, diff_dst_iter_, iter_dir, lay_dir,
                dir_val, n_dir, true, direction == mkldnn_bidirectional_concat);

        // We run the grid of computation
        for (int j = n_layer - 1; j >= 0; j--) {
            for (int i = 0; i < n_iter; i++) {
                int iter = (iter_dir == left2right) ? i + 1 : n_iter - i;
                int prev_iter = (iter_dir == left2right) ? iter - 1 : iter + 1;
                int lay = j + 1;
                int prev_lay = lay + 1;

                int ws_iter = (iter_dir == left2right) ? iter : iter;
                int ws_prev_iter
                        = (iter_dir == left2right) ? iter + 1 : iter - 1;

                rnn_cell_bwd(alg, f, sic, slc, dic, wc, batch, n_gates,
                        &wsb(lay, dir_val, iter, X, 0, 0),
                        &wsb(lay, dir_val, iter, H, 0, 0),
                        &wsb(lay, dir_val, iter, C, 0, 0),
                        &diff_weights_layer(lay - 1, dir_val, 0, 0),
                        &diff_weights_iter(lay - 1, dir_val, 0, 0),
                        &diff_bias(lay - 1, dir_val, 0, 0), b_gates,
                        &ws(lay - 1, dir_val, ws_iter, H, 0, 0),
                        &ws(lay, dir_val, ws_prev_iter, H, 0, 0),
                        &ws(lay, dir_val, ws_prev_iter, C, 0, 0),
                        &weights_layer(lay - 1, dir_val, 0, 0),
                        &weights_iter(lay - 1, dir_val, 0, 0),
                        &bias(lay - 1, dir_val, 0, 0),
                        &ws(lay, dir_val, ws_iter, H, 0, 0),
                        &ws(lay, dir_val, ws_iter, C, 0, 0),
                        &gates(lay - 1, dir_val, ws_iter - 1, 0, 0, 0),
                        &wsb(prev_lay, dir_val, iter, X, 0, 0),
                        &wsb(lay, dir_val, prev_iter, H, 0, 0),
                        &wsb(lay, dir_val, prev_iter, C, 0, 0),
                        ws_local_);
            }
        }

        // Finally we copy the results to the result buffers
        copy_res(alg, sic, slc, dic, dlc, wc, batch, n_layer, n_iter, n_states,
                diff_src_iter_, diff_src_layer_, wsb_, direction, iter_dir,
                lay_dir, dir_val, n_dir, action, true);
    };

    switch (direction) {
    case mkldnn_unidirectional_left2right:
        process_direction(right2left, top2bottom, 0, action_copy);
        break;
    case mkldnn_unidirectional_right2left:
        process_direction(left2right, top2bottom, 0, action_copy);
        break;
    case mkldnn_bidirectional_sum:
        process_direction(right2left, top2bottom, 0, action_copy);
        process_direction(left2right, top2bottom, 1, action_sum);
        break;
    case mkldnn_bidirectional_concat:
        process_direction(right2left, top2bottom, 0, action_copy);
        process_direction(left2right, top2bottom, 1, action_sum);
        break;
    default: assert("unknown direction"); break;
    }

    delete[] wsb_;
    delete[] b_gates;
    delete[] ws_local_;
}

void compute_ref_bwd(const rnn_prb_t *p, dnn_mem_t &input_m,
        dnn_mem_t &states_m, dnn_mem_t &diff_last_layer_m,
        dnn_mem_t &diff_last_iteration_m, dnn_mem_t &weights_input_m,
        dnn_mem_t &weights_states_m, dnn_mem_t &bias_m,
        dnn_mem_t &dst_last_layer_m, dnn_mem_t &dst_last_iteration_m,
        dnn_mem_t &dst_diff_input_m, dnn_mem_t &dst_diff_states_m,
        dnn_mem_t &dst_diff_weights_input_m,
        dnn_mem_t &dst_diff_weights_states_m, dnn_mem_t &dst_diff_bias_m,
        mkldnn_rnn_direction_t direction) {
    // !! TODO: add support of strides

    assert(direction == mkldnn_unidirectional_left2right
            || direction == mkldnn_unidirectional_right2left
            || direction == mkldnn_bidirectional_sum
            || direction == mkldnn_bidirectional_concat);

    assert(p->dlc == p->dic);
    int wc = max(p->sic, max(p->slc, p->dic));
    int ws_size = (p->n_layer + 2) * p->n_directions() * (p->n_iter + 2)
            * p->n_states() * p->mb * wc;
    auto *ws = new float[ws_size];
    init_buffer(ws, ws_size, -55.); // ??!! Temporary. For debug.
    int gates_size = p->n_layer * p->n_directions() * p->n_iter * p->mb
            * p->n_gates() * p->dic;
    auto *gates = new float[gates_size];

    rnn_linear_fwd(p, direction, (float *)states_m, (float *)input_m,
            (float *)weights_input_m, (float *)weights_states_m,
            (float *)bias_m, (float *)dst_last_iteration_m,
            (float *)dst_last_layer_m, ws, gates);

    rnn_linear_bwd(p, direction, (float *)diff_last_iteration_m,
            (float *)diff_last_layer_m, (float *)weights_input_m,
            (float *)weights_states_m, (float *)bias_m,
            (float *)dst_diff_states_m, (float *)dst_diff_input_m,
            (float *)dst_diff_weights_input_m,
            (float *)dst_diff_weights_states_m, (float *)dst_diff_bias_m, ws,
            gates);

    delete[] ws;
}

} // namespace rnn
