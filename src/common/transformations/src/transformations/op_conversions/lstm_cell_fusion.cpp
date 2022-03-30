// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/lstm_cell_fusion.hpp"

#include <memory>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

using namespace ov;

namespace {
    std::string get_activation_name(const std::shared_ptr<ov::Node>& activation) {
        auto sigmoid = std::dynamic_pointer_cast<opset8::Sigmoid>(activation);
        if (sigmoid) {
            return "sigmoid";
        }

        auto tanh = std::dynamic_pointer_cast<opset8::Tanh>(activation);
        if (tanh) {
            return "tanh";
        }

        auto relu = std::dynamic_pointer_cast<opset8::Relu>(activation);
        if (relu) {
            return "relu";
        }

        return "";
    }

    void get_hidden_size(const std::shared_ptr<ov::Node>& node, int64_t dim_idx, int64_t& hidden_size, int64_t div) {
        if (hidden_size != -1)
            return;

        auto partial_sh = node->get_output_partial_shape(0);
        if (partial_sh.rank().is_static() && partial_sh.rank().get_length() > dim_idx && partial_sh[dim_idx].is_static()
            && partial_sh[dim_idx].get_length() % div == 0) {
            hidden_size = partial_sh[dim_idx].get_length() / div;
        }
    }
}

ov::pass::LSTMCellFusion::LSTMCellFusion() {
    MATCHER_SCOPE(LSTMCellFusion);
    auto X = pattern::any_input();
    auto H_t = pattern::any_input();
    auto C_t = pattern::any_input();
    auto W = pattern::any_input();
    auto R = pattern::any_input();
    auto B = pattern::any_input();
    
    auto Xt_W = std::make_shared<opset8::MatMul>(X, W, false, true);
    auto Ht_R = std::make_shared<opset8::MatMul>(H_t, R, false, true);
    auto add = std::make_shared<opset8::Add>(Ht_R, B);
    auto XHB = std::make_shared<opset8::Add>(Xt_W, add);

    auto axis_node = ov::opset8::Constant::create(element::i64, Shape{}, {1});
    auto split = std::make_shared<opset8::Split>(XHB, axis_node, 4);
    Output<Node> f = split->output(0);
    Output<Node> i = split->output(1);
    Output<Node> c = split->output(2);
    Output<Node> o = split->output(3);

    auto f_t = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({f});
    auto i_t = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({i});
    auto c_t = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({c});
    auto o_t = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({o});

    auto mul1 = std::make_shared<opset8::Multiply>(f_t, C_t);
    auto mul2 = std::make_shared<opset8::Multiply>(i_t, c_t);
    auto out_C = std::make_shared<opset8::Add>(mul1, mul2);

    auto hC = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({out_C});
    auto out_H = std::make_shared<opset8::Multiply>(o_t, hC);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto activation = m.get_match_root();
        std::string activation_name = get_activation_name(activation);
        if (activation_name.empty()) {
            return false;
        }

        auto pattern_map = m.get_pattern_map();
        std::string f1_activation_name = get_activation_name(pattern_map.at(f_t));
        std::string f2_activation_name = get_activation_name(pattern_map.at(i_t));
        std::string g_activation_name = get_activation_name(pattern_map.at(c_t));
        std::string f3_activation_name = get_activation_name(pattern_map.at(o_t));
        std::string h_activation_name = get_activation_name(pattern_map.at(hC));

        if (g_activation_name.empty() || f1_activation_name.empty() || f2_activation_name.empty()
            || f3_activation_name.empty() || h_activation_name.empty() || f2_activation_name != f1_activation_name ||
            f2_activation_name != f3_activation_name) {
            return false;
        }

        auto x = pattern_map.at(X);
        auto h = pattern_map.at(H_t);
        auto c = pattern_map.at(C_t);
        auto w = pattern_map.at(W);
        auto r = pattern_map.at(R);
        auto b = pattern_map.at(B);
        int64_t hidden_size = -1;
        get_hidden_size(h, 1, hidden_size, 1);
        get_hidden_size(c, 1, hidden_size, 1);
        get_hidden_size(w, 0, hidden_size, 4);
        get_hidden_size(r, 0, hidden_size, 4);
        get_hidden_size(r, 1, hidden_size, 1);
        get_hidden_size(b, 0, hidden_size, 4);
        if (hidden_size == -1)
            return false;

        auto rnn_cell = std::make_shared<opset8::LSTMCell>(
                x, h, c, w, r, b, hidden_size, std::vector<std::string>{f1_activation_name, g_activation_name,
                                                                        h_activation_name});

        for (const auto& target : activation->get_output_target_inputs(0)) {
            target.replace_source_output(rnn_cell->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(out_H, matcher_name);
    register_matcher(m, callback);
}
