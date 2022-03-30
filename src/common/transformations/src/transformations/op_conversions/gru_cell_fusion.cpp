// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gru_cell_fusion.hpp"

#include <memory>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"
#include "transformations/utils/utils.hpp"

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

ov::pass::GRUCellFusion::GRUCellFusion() {
    MATCHER_SCOPE(GRUCellFusion);
    auto X = pattern::any_input();
    auto H_t = pattern::any_input();
    auto W = pattern::any_input();
    auto R = pattern::any_input();
    auto B = pattern::any_input();

    auto Xt_W = std::make_shared<opset8::MatMul>(X, W, false, true);
    auto Ht_R = std::make_shared<opset8::MatMul>(H_t, R, false, true);
    auto axis_0 = ov::opset8::Constant::create(element::i64, Shape{}, {0});
    auto axis_1 = ov::opset8::Constant::create(element::i64, Shape{}, {1});
    auto Xt_W_zrh = std::make_shared<opset8::Split>(Xt_W, axis_1, 3);
    auto R_zrh = std::make_shared<opset8::Split>(R, axis_0, 3);
    auto Ht_R_zrh = std::make_shared<opset8::Split>(Ht_R, axis_1, 3);
    auto biases_zrh = std::make_shared<opset8::Split>(B, axis_0, 4);

    auto add_z_1 = std::make_shared<opset8::Add>(Ht_R_zrh->output(0), biases_zrh->output(0));
    auto add_z_2 = std::make_shared<opset8::Add>(Xt_W_zrh->output(0), add_z_1);
    auto add_r_1 = std::make_shared<opset8::Add>(Ht_R_zrh->output(1), biases_zrh->output(1));
    auto add_r_2 = std::make_shared<opset8::Add>(Xt_W_zrh->output(1), add_r_1);

    auto z_t = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({add_z_2});
    auto r_t = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({add_r_2});

    std::shared_ptr<Node> _h;
    auto Ht_Rh_Rbh = std::make_shared<opset8::Add>(Ht_R_zrh->output(2), biases_zrh->output(3));
    auto mul_h_1 = std::make_shared<opset8::Multiply>(r_t, Ht_Rh_Rbh);
    auto add_h_1 = std::make_shared<opset8::Add>(mul_h_1, biases_zrh->output(2));
    _h = std::make_shared<opset8::Add>(Xt_W_zrh->output(2), add_h_1);

    auto h_t = ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({_h});

    auto one = opset8::Constant::create(z_t->get_element_type(), Shape{1}, {1.f});
    auto sub = std::make_shared<opset8::Subtract>(one, z_t);
    auto mul_1 = std::make_shared<opset8::Multiply>(sub, h_t);
    auto mul_2 = std::make_shared<opset8::Multiply>(z_t, H_t);
    auto out = std::make_shared<opset8::Add>(mul_1, mul_2);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto add = m.get_match_root();
        if (!ov::as_type_ptr<opset8::Add>(add)) {
            return false;
        }

        auto pattern_map = m.get_pattern_map();

        std::string g_activation_name = get_activation_name(pattern_map.at(h_t));
        std::string f1_activation_name = get_activation_name(pattern_map.at(r_t));
        std::string f2_activation_name = get_activation_name(pattern_map.at(z_t));

        if (g_activation_name.empty() || f1_activation_name.empty() || f2_activation_name.empty() ||
                f2_activation_name != f1_activation_name) {
            return false;
        }

        auto x = pattern_map.at(X);
        auto h = pattern_map.at(H_t);
        auto w = pattern_map.at(W);
        auto r = pattern_map.at(R);
        auto b = pattern_map.at(B);
        int64_t hidden_size = -1;
        get_hidden_size(h, 1, hidden_size, 1);
        get_hidden_size(w, 0, hidden_size, 3);
        get_hidden_size(r, 0, hidden_size, 3);
        get_hidden_size(r, 1, hidden_size, 1);
        get_hidden_size(b, 0, hidden_size, 4);
        if (hidden_size == -1)
            return false;

        auto gru_cell = std::make_shared<opset8::GRUCell>(
                x, h, w, r, b, hidden_size, std::vector<std::string>{g_activation_name,
                                                                     f1_activation_name});

        for (const auto& target : pattern_map.at(out)->get_output_target_inputs(0)) {
            target.replace_source_output(gru_cell->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(out, matcher_name);
    register_matcher(m, callback);
}
