// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/rnn_cell_fusion.hpp"

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/utils/utils.hpp"

#include <iostream>

// Input (12, 6)
// batch_size (22)
// Num of RNNs (4)

static std::string get_activation_name(const std::shared_ptr<ov::Node>& node) {
    std::string name = node->get_type_name();
    name[0] = std::tolower(name[0]);
    return name;
}

// TODO: check if more complicated logic is needed
// I was told about some 1 (ones) are present
static std::size_t get_hidden_size_from_bias_shape(const ov::Shape& shape) {
    if (shape.empty()) {
        return 0;
    }

    return shape.at(0);
}

static std::shared_ptr<ov::Node> transpose_input(const std::shared_ptr<ov::Node>& node, bool transposed) {
   std::shared_ptr<ov::Node> tail = node;
   if (!transposed) {
       auto transpose_order = std::make_shared<ov::op::v0::Constant>(ov::element::u32, ov::Shape{2}, ov::Shape{1, 0});
       tail = std::make_shared<ov::op::v1::Transpose>(tail, transpose_order);
   }

   return tail;
}

ov::pass::RNNCellTfKerasFusion::RNNCellTfKerasFusion() {
    MATCHER_SCOPE(RNNCellTfKerasFusion);

    auto X_label = pattern::any_input();
    auto W_label = pattern::any_input();
    auto matmul1_label = pattern::wrap_type<op::v0::MatMul>({X_label, W_label});

    auto B_label = pattern::any_input();
    auto add_label_1 = pattern::wrap_type<op::v1::Add>({matmul1_label, B_label});

    auto Ht_label = pattern::any_input();
    auto R_label = pattern::any_input();
    auto matmul2_label = pattern::wrap_type<op::v0::MatMul>({Ht_label, R_label});

    auto add_label_2 = pattern::wrap_type<op::v1::Add>({add_label_1, matmul2_label});

    auto activation_func_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({add_label_2});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        std::cout << "__________MATCHED__________" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& matmul_1 = pattern_map.at(matmul1_label);
        const auto& add_1 = pattern_map.at(add_label_1);

        const auto& matmul_2 = pattern_map.at(matmul2_label);
        const auto& add_2 = pattern_map.at(add_label_2);

        const auto& X = pattern_map.at(X_label);
        const auto& Ht = pattern_map.at(Ht_label);
        const auto& W = pattern_map.at(W_label);
        const auto& R = pattern_map.at(R_label);
        const auto& B = pattern_map.at(B_label);
        std::size_t hidden_size = get_hidden_size_from_bias_shape(B.get_shape());
        auto act_func = pattern_map.at(activation_func_label);

        bool is_W_transposed = ov::as_type_ptr<op::v0::MatMul>(matmul_1.get_node_shared_ptr())->get_transpose_b();
        std::shared_ptr<Node> W_input = transpose_input(W.get_node_shared_ptr(), is_W_transposed);

        bool is_R_transposed = ov::as_type_ptr<op::v0::MatMul>(matmul_2.get_node_shared_ptr())->get_transpose_b();
        std::shared_ptr<Node> R_input = transpose_input(R.get_node_shared_ptr(), is_R_transposed);

        const std::string& act_func_name = get_activation_name(act_func.get_node_shared_ptr());

        auto rnn_cell = std::make_shared<ov::op::v0::RNNCell>(
            X,
            Ht,
            W_input,
            R_input,
            B,
            hidden_size,
            std::vector<std::string>{act_func_name}
        );

        rnn_cell->set_friendly_name(m.get_match_root()->get_friendly_name());
        act_func.replace(rnn_cell->output(0));

        if (transformation_callback(rnn_cell)) {
            std::cout << "transformation_callback() == false" << std::endl;
            return false;
        }

        ov::copy_runtime_info(
            {
                X.get_node_shared_ptr(),
                Ht.get_node_shared_ptr(),
                W.get_node_shared_ptr(),
                R.get_node_shared_ptr(),
                B.get_node_shared_ptr(),
                matmul_1.get_node_shared_ptr(),
                matmul_2.get_node_shared_ptr(),
                add_1.get_node_shared_ptr(),
                add_2.get_node_shared_ptr(),
            },
            {
                W_input,
                R_input,
                rnn_cell
            }
        );

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(activation_func_label, matcher_name);
    this->register_matcher(m, callback);
}