// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/lstm_cell_decomposition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::LSTMCellDecomposition::LSTMCellDecomposition() {
    MATCHER_SCOPE(LSTMCellDecomposition);
    auto any_lstm = pattern::wrap_type<ov::op::v0::LSTMCell, ov::op::v4::LSTMCell>();

    matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& m) {
        auto lstm_cell = ov::as_type_ptr<op::util::RNNCellBase>(m.get_match_root());
        if (!lstm_cell || transformation_callback(lstm_cell)) {
            return false;
        }
        const Output<Node>& X = lstm_cell->input_value(0);
        const Output<Node>& H_t = lstm_cell->input_value(1);
        const Output<Node>& C_t = lstm_cell->input_value(2);
        const Output<Node>& W = lstm_cell->input_value(3);
        const Output<Node>& R = lstm_cell->input_value(4);
        const Output<Node>& bias = lstm_cell->input_value(5);

        // Xt*(W^T)
        auto Xt_W = std::make_shared<ov::op::v0::MatMul>(X, W, false, true);
        // Ht-1*(R^T)
        auto Ht_R = std::make_shared<ov::op::v0::MatMul>(H_t, R, false, true);
        // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
        auto add = std::make_shared<ov::op::v1::Add>(Ht_R, bias);
        auto XHB = std::make_shared<ov::op::v1::Add>(Xt_W, add);

        auto axis_node = ov::op::v0::Constant::create(element::u64, Shape{}, {1});
        auto split = std::make_shared<ov::op::v1::Split>(XHB, axis_node, 4);
        Output<Node> f = split->output(0);
        Output<Node> i = split->output(1);
        Output<Node> c = split->output(2);
        Output<Node> o = split->output(3);

        auto clip = lstm_cell->get_clip();
        if (clip > 0.f) {
            auto clamp_f = std::make_shared<ov::op::v0::Clamp>(f, -clip, clip);
            auto clamp_i = std::make_shared<ov::op::v0::Clamp>(i, -clip, clip);
            auto clamp_c = std::make_shared<ov::op::v0::Clamp>(c, -clip, clip);
            auto clamp_o = std::make_shared<ov::op::v0::Clamp>(o, -clip, clip);
            f = clamp_f;
            i = clamp_i;
            c = clamp_c;
            o = clamp_o;
            ov::copy_runtime_info(lstm_cell, {clamp_f, clamp_i, clamp_c, clamp_o});
        }

        // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)
        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)
        auto f_t = ov::op::util::activation(lstm_cell->get_activations()[0], f);
        auto i_t = ov::op::util::activation(lstm_cell->get_activations()[0], i);
        auto c_t = ov::op::util::activation(lstm_cell->get_activations()[1], c);
        auto o_t = ov::op::util::activation(lstm_cell->get_activations()[0], o);

        // Ct = ft (.) Ct-1 + it (.) ct
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(f_t, C_t);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(i_t, c_t);
        auto out_C = std::make_shared<ov::op::v1::Add>(mul1, mul2);

        // H = ot (.) h(Ct)
        auto hC = ov::op::util::activation(lstm_cell->get_activations()[2], out_C);
        auto out_H = std::make_shared<ov::op::v1::Multiply>(o_t, hC);

        out_H->set_friendly_name(lstm_cell->get_friendly_name() + ".0");
        out_C->set_friendly_name(lstm_cell->get_friendly_name() + ".1");
        ov::copy_runtime_info(
            lstm_cell,
            {Xt_W, Ht_R, add, split, mul1, mul2, out_H, hC, out_C, axis_node, XHB, f_t, i_t, c_t, o_t});
        ov::replace_node(lstm_cell, {out_H->output(0), out_C->output(0)});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(any_lstm, matcher_name);
    register_matcher(m, callback);
}
