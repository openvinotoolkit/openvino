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

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::types;
using namespace mkldnn::impl::utils;

namespace {
memory_desc_t copy_maybe_null(const memory_desc_t *md) {
    return md ? *md : zero_md();
}

rnn_desc_t zero_rnn_desc() {
    auto rd = rnn_desc_t();
    rd.src_layer_desc = zero_md();
    rd.src_iter_desc = zero_md();
    rd.weights_layer_desc = zero_md();
    rd.weights_iter_desc = zero_md();
    rd.bias_desc = zero_md();
    rd.dst_layer_desc = zero_md();
    rd.dst_iter_desc = zero_md();
    rd.diff_src_layer_desc = zero_md();
    rd.diff_src_iter_desc = zero_md();
    rd.diff_weights_layer_desc = zero_md();
    rd.diff_weights_iter_desc = zero_md();
    rd.diff_bias_desc = zero_md();
    rd.diff_dst_layer_desc = zero_md();
    rd.diff_dst_iter_desc = zero_md();
    return rd;
}
}

/* Public C Api */

status_t mkldnn_rnn_cell_desc_init(rnn_cell_desc_t *rnn_cell_desc,
        mkldnn_alg_kind_t cell_kind, mkldnn_alg_kind_t act_f,
        unsigned int flags, float alpha, float clipping) {
    using namespace mkldnn::impl::alg_kind;

    bool args_ok = true
            && one_of(cell_kind, vanilla_rnn, vanilla_lstm, vanilla_gru,
                    gru_linear_before_reset)
            && IMPLICATION(cell_kind == vanilla_rnn,
                    one_of(act_f, eltwise_relu, eltwise_tanh, eltwise_logistic));
    if (!args_ok)
        return status::invalid_arguments;

    auto rcd = mkldnn_rnn_cell_desc_t();

    rcd.cell_kind = cell_kind;
    rcd.activation_kind = act_f;
    rcd.flags = flags;
    rcd.alpha = rcd.flags & mkldnn_rnn_cell_with_relu ? alpha : 0;
    rcd.clipping = rcd.flags & mkldnn_rnn_cell_with_clipping ? clipping : 0;

    *rnn_cell_desc = rcd;

    return status::success;
}

int mkldnn_rnn_cell_get_gates_count(const rnn_cell_desc_t *rnn_cell_desc) {
    switch (rnn_cell_desc->cell_kind) {
    case mkldnn::impl::alg_kind::vanilla_rnn: return 1;
    case mkldnn::impl::alg_kind::vanilla_gru: return 3;
    case mkldnn::impl::alg_kind::gru_linear_before_reset: return 3;
    case mkldnn::impl::alg_kind::vanilla_lstm: return 4;
    default: assert(!"unknown cell kind"); return 0;
    }
    return 0;
}

int mkldnn_rnn_cell_get_states_count(const rnn_cell_desc_t *rnn_cell_desc) {
    switch (rnn_cell_desc->cell_kind) {
    case mkldnn::impl::alg_kind::vanilla_rnn: return 1;
    case mkldnn::impl::alg_kind::vanilla_gru: return 1;
    case mkldnn::impl::alg_kind::gru_linear_before_reset: return 1;
    case mkldnn::impl::alg_kind::vanilla_lstm: return 2;
    default: assert(!"unknown cell kind"); return 0;
    }
    return 0;
}

status_t MKLDNN_API mkldnn_rnn_forward_desc_init(mkldnn_rnn_desc_t *rnn_desc,
        prop_kind_t prop_kind, const rnn_cell_desc_t *rnn_cell_desc,
        const rnn_direction_t direction, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc,
        const memory_desc_t *dst_iter_desc) {
    bool args_ok = true && rnn_cell_desc != nullptr
            && !any_null(src_layer_desc, weights_layer_desc, weights_iter_desc,
                       dst_layer_desc);
    if (!args_ok)
        return invalid_arguments;

    int DIC = 0, L = 0;
    if (weights_layer_desc->ndims) {
        DIC = weights_layer_desc->dims[4];
        L = weights_layer_desc->dims[0];
    } else if (weights_iter_desc->ndims) {
        DIC = weights_iter_desc->dims[4];
        L = weights_iter_desc->dims[0];
    } else {
        assert(!"cannot query cell state size");
        return unimplemented;
    }

    const int D = one_of(direction, mkldnn_unidirectional_left2right,
                          mkldnn_unidirectional_right2left) ?
            1 :
            2;
    const int DLC = (direction == mkldnn_bidirectional_concat ? 2 : 1) * DIC;

    args_ok = args_ok && D == weights_layer_desc->dims[1]
            && D == weights_iter_desc->dims[1]
            && DIC == weights_layer_desc->dims[4]
            && DIC == weights_iter_desc->dims[4]
            && DLC == dst_layer_desc->dims[2] && L == weights_iter_desc->dims[0]
            && IMPLICATION(!is_zero_md(dst_iter_desc), true
                               && DIC == dst_iter_desc->dims[4]
                               && L == dst_iter_desc->dims[0])
            && IMPLICATION(!is_zero_md(bias_desc), L == bias_desc->dims[0])
            && IMPLICATION(
                       !is_zero_md(src_iter_desc), L == src_iter_desc->dims[0])
            && IMPLICATION(rnn_cell_desc->cell_kind == alg_kind::vanilla_gru,
                       DIC == weights_iter_desc->dims[2]);
    if (!args_ok)
        return invalid_arguments;

    mkldnn_rnn_desc_t rd = zero_rnn_desc();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_desc = *rnn_cell_desc;
    rd.direction = direction;
    rd.src_layer_desc = copy_maybe_null(src_layer_desc);
    rd.src_iter_desc = copy_maybe_null(src_iter_desc);
    rd.weights_layer_desc = copy_maybe_null(weights_layer_desc);
    rd.weights_iter_desc = copy_maybe_null(weights_iter_desc);
    rd.bias_desc = copy_maybe_null(bias_desc);
    rd.dst_layer_desc = copy_maybe_null(dst_layer_desc);
    rd.dst_iter_desc = copy_maybe_null(dst_iter_desc);

    *rnn_desc = rd;

    return success;
}

status_t MKLDNN_API mkldnn_rnn_backward_desc_init(mkldnn_rnn_desc_t *rnn_desc,
        prop_kind_t prop_kind, const rnn_cell_desc_t *rnn_cell_desc,
        const rnn_direction_t direction, const memory_desc_t *src_layer_desc,
        const memory_desc_t *src_iter_desc,
        const memory_desc_t *weights_layer_desc,
        const memory_desc_t *weights_iter_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_layer_desc, const memory_desc_t *dst_iter_desc,
        const memory_desc_t *diff_src_layer_desc,
        const memory_desc_t *diff_src_iter_desc,
        const memory_desc_t *diff_weights_layer_desc,
        const memory_desc_t *diff_weights_iter_desc,
        const memory_desc_t *diff_bias_desc,
        const memory_desc_t *diff_dst_layer,
        const memory_desc_t *diff_dst_iter_desc) {
    bool args_ok = true
            && !any_null(src_layer_desc, weights_layer_desc, weights_iter_desc,
                       dst_layer_desc, diff_src_layer_desc,
                       diff_weights_layer_desc, diff_weights_iter_desc,
                       diff_dst_layer);
    if (!args_ok)
        return invalid_arguments;

    int DIC = 0, L = 0;
    if (weights_layer_desc->ndims) {
        DIC = weights_layer_desc->dims[4];
        L = weights_layer_desc->dims[0];
    } else if (weights_iter_desc->ndims) {
        DIC = weights_iter_desc->dims[4];
        L = weights_iter_desc->dims[0];
    } else {
        assert(!"cannot query cell state size");
        return unimplemented;
    }

    auto xnor_md = [=](const memory_desc_t *a_md, const memory_desc_t *b_md) {
        return is_zero_md(a_md) == is_zero_md(b_md);
    };

    args_ok = args_ok && xnor_md(bias_desc, diff_bias_desc)
            && xnor_md(dst_iter_desc, diff_dst_iter_desc)
            && xnor_md(src_iter_desc, diff_src_iter_desc);
    if (!args_ok)
        return invalid_arguments;

    int D = one_of(direction, mkldnn_unidirectional_left2right,
                    mkldnn_unidirectional_right2left) ?
            1 :
            2;
    int DLC = (direction == mkldnn_bidirectional_concat ? 2 : 1) * DIC;

    args_ok = args_ok && D == weights_layer_desc->dims[1]
            && D == weights_iter_desc->dims[1]
            && DIC == weights_layer_desc->dims[4]
            && DIC == weights_iter_desc->dims[4]
            && DLC == dst_layer_desc->dims[2] && L == weights_iter_desc->dims[0]
            && IMPLICATION(!is_zero_md(dst_iter_desc), true
                               && DIC == dst_iter_desc->dims[4]
                               && L == dst_iter_desc->dims[0])
            && IMPLICATION(!is_zero_md(bias_desc), L == bias_desc->dims[0])
            && IMPLICATION(
                       !is_zero_md(src_iter_desc), L == src_iter_desc->dims[0])
            && IMPLICATION(rnn_cell_desc->cell_kind == alg_kind::vanilla_gru,
                       DIC == weights_iter_desc->dims[2]);
    if (!args_ok)
        return invalid_arguments;

    mkldnn_rnn_desc_t rd = zero_rnn_desc();

    rd.primitive_kind = primitive_kind::rnn;
    rd.prop_kind = prop_kind;
    rd.cell_desc = *rnn_cell_desc;
    rd.direction = direction;

    rd.src_layer_desc = copy_maybe_null(src_layer_desc);
    rd.src_iter_desc = copy_maybe_null(src_iter_desc);
    rd.weights_layer_desc = copy_maybe_null(weights_layer_desc);
    rd.weights_iter_desc = copy_maybe_null(weights_iter_desc);
    rd.bias_desc = copy_maybe_null(bias_desc);
    rd.dst_layer_desc = copy_maybe_null(dst_layer_desc);
    rd.dst_iter_desc = copy_maybe_null(dst_iter_desc);
    rd.diff_src_layer_desc = copy_maybe_null(diff_src_layer_desc);
    rd.diff_src_iter_desc = copy_maybe_null(diff_src_iter_desc);
    rd.diff_weights_layer_desc = copy_maybe_null(diff_weights_layer_desc);
    rd.diff_weights_iter_desc = copy_maybe_null(diff_weights_iter_desc);
    rd.diff_bias_desc = copy_maybe_null(diff_bias_desc);
    rd.diff_dst_layer_desc = copy_maybe_null(diff_dst_layer);
    rd.diff_dst_iter_desc = copy_maybe_null(diff_dst_iter_desc);

    *rnn_desc = rd;

    return success;
}
