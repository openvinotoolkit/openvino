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

    auto XH = std::make_shared<opset8::Concat>(OutputVector{X, H_t}, 1);
    auto matmul_WR_zr = std::make_shared<opset8::MatMul>(XH, pattern::any_input(), false, true);
    auto add_B_zr = std::make_shared<opset8::Add>(matmul_WR_zr, B);
    auto zr_t = std::make_shared<opset8::Sigmoid>(add_B_zr); //ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({_h});

    auto axis_1 = ov::opset8::Constant::create(element::i64, Shape{}, {1});
    auto z_r = std::make_shared<opset8::Split>(zr_t, axis_1, 2);

    auto XH_t = std::make_shared<opset8::Multiply>(z_r->output(1), pattern::any_input());
    auto XHr = std::make_shared<opset8::Concat>(OutputVector{pattern::any_input(), XH_t}, 1);

    auto matmul_WR_h = std::make_shared<opset8::MatMul>(XHr, pattern::any_input());
    auto add_B_h = std::make_shared<opset8::Add>(matmul_WR_h, pattern::any_input());
    auto h_t = std::make_shared<opset8::Tanh>(add_B_h); // ov::pass::pattern::wrap_type<opset8::Sigmoid, opset8::Tanh, opset8::Relu>({_h});

    auto one = opset8::Constant::create(element::f32, Shape{1}, {1.f});
    auto sub = std::make_shared<opset8::Subtract>(one, z_r->output(0)); //
    auto mul_1 = std::make_shared<opset8::Multiply>(h_t, sub);
    auto mul_2 = std::make_shared<opset8::Multiply>(pattern::any_input(), z_r->output(1));
    auto out = std::make_shared<opset8::Add>(mul_2, mul_1);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        std::cout << "XXXXXXXXXXXXXXXXXXXXXXx" << "CALLBACK" << std::endl;
        auto add = m.get_match_root();
        if (!ov::as_type_ptr<opset8::Add>(add)) {
            return false;
        }

        auto pattern_map = m.get_pattern_map();
        std::string f_activation_name = get_activation_name(pattern_map.at(zr_t));
        std::string g_activation_name = get_activation_name(pattern_map.at(h_t));

        if (g_activation_name.empty() || f_activation_name.empty()) {
            return false;
        }

        auto x = pattern_map.at(X);
        auto h = pattern_map.at(H_t);
        auto WR_zr = pattern_map.at(matmul_WR_zr);
        auto B_zr = pattern_map.at(add_B_zr);
        auto WR_h = pattern_map.at(matmul_WR_h);
        auto B_h = pattern_map.at(add_B_h);
        int64_t hidden_size = -1;
        get_hidden_size(h, 1, hidden_size, 1);
        if (hidden_size == -1)
            return false;

        auto axis_0 = ov::opset8::Constant::create(element::i64, Shape{}, {0});
        auto WR_z_r = std::make_shared<ov::opset8::Split>(WR_zr->input_value(1), axis_0, 2);
        auto B_z_r = std::make_shared<ov::opset8::Split>(B_zr->input_value(1), axis_1, 2);

        auto B = std::make_shared<ov::opset8::Concat>(OutputVector{B_z_r->output(0), B_z_r->output(1), B_h->input_value(1)},
                                                      1);
        auto WR = std::make_shared<ov::opset8::Concat>(OutputVector{WR_z_r->output(0), WR_z_r->output(1), WR_h->input_value(1)},
                                                       0);
        auto axis_1 = ov::opset8::Constant::create(element::i64, Shape{}, {1});
        auto W_R = std::make_shared<ov::opset8::Split>(WR, axis_1, 2);
        auto gru_cell = std::make_shared<opset8::GRUCell>(
                x,
                h,
                W_R->output(0),
                W_R->output(1),
                std::make_shared<opset8::Squeeze>(B, axis_0),
                hidden_size,
                std::vector<std::string>{g_activation_name,
                                         f_activation_name});

        for (const auto& target : pattern_map.at(out)->get_output_target_inputs(0)) {
            target.replace_source_output(gru_cell->output(0));
        }

        auto unused = pattern_map.at(mul_2)->input_value(1);
        auto unused_2 = pattern_map.at(XH_t)->input_value(1);
        auto result = std::make_shared<opset8::Result>(unused);
        auto result_2 = std::make_shared<opset8::Result>(unused_2);
        gru_cell->add_control_dependency(result);
        gru_cell->add_control_dependency(result_2);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(out, matcher_name);
    register_matcher(m, callback);
}
