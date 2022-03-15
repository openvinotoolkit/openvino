// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/rnn_cell_decomposition.hpp"

#include <memory>
#include <ngraph/op/util/activation_functions.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

ngraph::pass::RNNCellDecomposition::RNNCellDecomposition() {
    MATCHER_SCOPE(RNNCellDecomposition);
    auto rnn_cell = ngraph::pattern::wrap_type<opset4::RNNCell>();
    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto rnn_cell = std::dynamic_pointer_cast<ngraph::opset4::RNNCell>(m.get_match_root());
        if (!rnn_cell || transformation_callback(rnn_cell)) {
            return false;
        }
        const Output<Node>& X = rnn_cell->input_value(0);
        const Output<Node>& H_t = rnn_cell->input_value(1);
        const Output<Node>& W = rnn_cell->input_value(2);
        const Output<Node>& R = rnn_cell->input_value(3);
        const Output<Node>& bias = rnn_cell->input_value(4);

        // Xt*(W^T)
        auto Xt_W = std::make_shared<opset4::MatMul>(X, W, false, true);
        // Ht-1*(R^T)
        auto Ht_R = std::make_shared<opset4::MatMul>(H_t, R, false, true);
        // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb
        auto add = std::make_shared<opset4::Add>(Ht_R, bias);
        auto i_t = std::make_shared<opset4::Add>(Xt_W, add);

        // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        auto clip = rnn_cell->get_clip();
        std::shared_ptr<Node> clamp = i_t;
        if (clip > 0.f) {
            clamp = std::make_shared<opset4::Clamp>(i_t, -clip, clip);
            ngraph::copy_runtime_info(rnn_cell, clamp);
        }
        auto out = ngraph::op::util::activation(rnn_cell->get_activations()[0], clamp);
        out->set_friendly_name(rnn_cell->get_friendly_name());
        ngraph::copy_runtime_info(rnn_cell, {Xt_W, Ht_R, add, i_t, out});
        ngraph::replace_node(rnn_cell, out);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnn_cell, matcher_name);
    register_matcher(m, callback);
}
