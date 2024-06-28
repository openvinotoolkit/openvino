// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_sdpa.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

using namespace ov::op;

ov::pass::MakeSDPA::MakeSDPA(std::shared_ptr<ov::Model> model) {
    MATCHER_SCOPE(MakeSDPA);

    auto Q_label = pattern::any_input();
    auto K_label = pattern::any_input();
    auto matmul_label = pattern::wrap_type<op::v0::MatMul>({Q_label, K_label});

    auto scale_label = pattern::any_input();
    auto divide_label = pattern::wrap_type<op::v1::Divide>({matmul_label, scale_label}); //TODO: figure out with divide

    auto select_label = pattern::wrap_type<op::v1::Select>({pattern::any_input(), divide_label, pattern::any_input()});

    auto add_label = pattern::wrap_type<op::v1::Add>({select_label, pattern::any_input()});
    auto add_label1 = pattern::wrap_type<op::v1::Add>({add_label, pattern::any_input()});

    auto softmax_label = pattern::wrap_type<op::v8::Softmax>({add_label1});
    auto V_label = pattern::any_input();
    auto final_matmul_label = pattern::wrap_type<op::v0::MatMul>({softmax_label, V_label});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &model](pattern::Matcher& m) {
        std::cout << "____" << matcher_name << "____" << std::endl;
        const auto& pattern_map = m.get_pattern_value_map();

        auto Q = pattern_map.at(Q_label);
        auto K = pattern_map.at(K_label);
        auto V = pattern_map.at(V_label);

        auto scale = pattern_map.at(scale_label);
        auto scale_converted = std::make_shared<op::v1::Divide>(op::v0::Constant::create(element::f32, Shape{}, {1}), scale);
        auto dummy_mask = op::v0::Constant::create(element::i32, Shape{}, {0});

        auto res_output = model->add_output(scale_converted->output(0));
        res_output.add_names({"ANDREW"});

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(Q, K, V, dummy_mask, scale_converted, true);
        sdpa->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::replace_node(m.get_match_root(), sdpa);

        if (transformation_callback(sdpa)) {
            std::cout << "transformation_callback() == false" << std::endl;
            return false;
        }

        ov::copy_output_runtime_info(
            {
                pattern_map.at(Q_label).get_node_shared_ptr(),
                pattern_map.at(K_label).get_node_shared_ptr(),
                pattern_map.at(V_label).get_node_shared_ptr(),
                pattern_map.at(matmul_label).get_node_shared_ptr(),
                scale.get_node_shared_ptr(),
                pattern_map.at(divide_label).get_node_shared_ptr(),
                pattern_map.at(select_label).get_node_shared_ptr(),
                pattern_map.at(add_label).get_node_shared_ptr(),
                pattern_map.at(add_label1).get_node_shared_ptr(),
                pattern_map.at(softmax_label).get_node_shared_ptr(),
                pattern_map.at(final_matmul_label).get_node_shared_ptr(),
            },
            {
                sdpa,
                scale_converted,
                dummy_mask
            }
        );

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(final_matmul_label, matcher_name);
    this->register_matcher(m, callback);
}