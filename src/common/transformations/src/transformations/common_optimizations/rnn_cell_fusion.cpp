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
#include "openvino/op/tanh.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <iostream>

ov::pass::RNNCellTfKerasFusion::RNNCellTfKerasFusion() {
    MATCHER_SCOPE(RNNCellTfKerasFusion);

    //right
    auto H = pattern::any_input();
    auto R = pattern::any_input();
    //think to check if R is transposed
    auto matmul_label_1 = pattern::wrap_type<op::v0::MatMul>({H, R});

    auto B = pattern::any_input(); //?
    auto add_label_1 = pattern::wrap_type<op::v1::Add>({B, matmul_label_1});

    //left
    auto X = pattern::any_input();
    auto W = pattern::any_input();
    //think to check if W is transposed
    auto matmul_label_2 = pattern::wrap_type<op::v0::MatMul>({X, W});

    auto add_label_2 = pattern::wrap_type<op::v1::Add>({add_label_1, matmul_label_2});

    auto activation_func_label = pattern::wrap_type<op::v0::Relu, op::v0::Sigmoid, op::v0::Tanh>({add_label_2});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        // const auto& pattern_map = m.get_pattern_value_map();
        std::cout << "__________MATCHED__________" << std::endl;
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(activation_func_label, matcher_name);
    this->register_matcher(m, callback);
}