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

ov::pass::RNNCellTfKerasFusion::RNNCellTfKerasFusion() {
    MATCHER_SCOPE(RNNCellTfKerasFusion);

    //right
    auto Ht_label = pattern::any_input();
    auto R_label = pattern::any_input();
    //think to check if R is transposed
    auto matmul1_label = pattern::wrap_type<op::v0::MatMul>({Ht_label, R_label});

    // B is Wb + Rb probably or something else combined
    auto B_label = pattern::any_input(); //?
    auto add_label_1 = pattern::wrap_type<op::v1::Add>({B_label, matmul1_label});

    //left
    auto X_label = pattern::any_input();
    auto W_label = pattern::any_input();
    //think to check if W is transposed
    auto matmul2_label = pattern::wrap_type<op::v0::MatMul>({X_label, W_label});

    auto add_label_2 = pattern::wrap_type<op::v1::Add>({add_label_1, matmul2_label});

    auto activation_func_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({add_label_2});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        std::cout << "__________MATCHED__________" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();

        //right
        // const auto& matmul_1 = pattern_map.at(matmul_label_1);
        // const auto& add_1 = pattern_map.at(add_label_1);

        //left
        // const auto& matmul_2 = pattern_map.at(matmul_label_2);
        // const auto& add_2 = pattern_map.at(add_label_2);

        const auto& X = pattern_map.at(X_label);
        const auto& Ht = pattern_map.at(Ht_label);
        const auto& W = pattern_map.at(W_label);
        auto matmul_2 = ov::as_type_ptr<op::v0::MatMul>(pattern_map.at(matmul1_label).get_node_shared_ptr());
        bool is_W_transposed = matmul_2->get_transpose_b();
        if (is_W_transposed) {
            std::cout << "W transposed" << std::endl;
        } else {
            std::cout << "W not transposed" << std::endl;
        }
        const auto& R = pattern_map.at(R_label);
        auto matmul_1 = ov::as_type_ptr<op::v0::MatMul>(pattern_map.at(matmul2_label).get_node_shared_ptr());
        bool is_R_transposed = matmul_1->get_transpose_b();
        if (is_R_transposed) {
            std::cout << "R transposed" << std::endl;
        } else {
            std::cout << "R not transposed" << std::endl;
        }
        const auto& B = pattern_map.at(B_label);

        // As I understood, this is num of RNN cells
        std::size_t hidden_size = get_hidden_size_from_bias_shape(B.get_shape()); //TODO find out what this is

        auto act_func = pattern_map.at(activation_func_label);
        const std::string& act_func_name = get_activation_name(act_func.get_node_shared_ptr());

        //TODO: check for transpose

        std::cout << "Hidden size : " << hidden_size << std::endl;

        auto rnn_cell = std::make_shared<ov::op::v0::RNNCell>(
            X,
            Ht,
            W,
            R,
            B,
            hidden_size,
            std::vector<std::string>{act_func_name}
        );

        //Ask about it
        rnn_cell->set_friendly_name(m.get_match_root()->get_friendly_name());
        act_func.replace(rnn_cell->output(0));

        // What is it?
        if (transformation_callback(rnn_cell)) {
            std::cout << "transformation_callback() == false" << std::endl;
            return false;
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(activation_func_label, matcher_name);
    this->register_matcher(m, callback);
}