// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multinomial.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::MultinomialRandomUniformFusion::MultinomialRandomUniformFusion() {
    MATCHER_SCOPE(MultinomialRandomUniformFusion);

    auto multinomial_pattern = ov::pass::pattern::wrap_type<ov::op::v13::Multinomial>(
        {ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto node = pattern_map.at(multinomial_pattern);
        const auto multinomial = ov::as_type_ptr<ov::op::v13::Multinomial>(node.get_node_shared_ptr());

        if (!multinomial)
            return false;

        if (multinomial->get_input_size() == 3) {
            return false;
        }

        // Insert RandomUniform
        auto random_samples = std::make_shared<ov::op::v8::RandomUniform>(
            node->input_value(0),
            ov::op::v0::Constant::create(node->get_input_element_type(0), ov::Shape{}, {0}),
            ov::op::v0::Constant::create(node->get_input_element_type(0), ov::Shape{}, {1}),
            node->get_input_element_type(0),
            node->get_global_seed(),
            node->get_op_seed());

        auto new_multinomial = std::make_shared<ov::op::v13::Multinomial>(node->input_value(0),
                                                                          node->input_value(1),
                                                                          random_samples,
                                                                          node->get_with_replacement(),
                                                                          node->get_log_probs(),
                                                                          node->get_global_seed(),
                                                                          node->get_op_seed());

        new_multinomial->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, {random_samples, new_multinomial});
        ov::replace_node(node, new_multinomial);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(multinomial_pattern, "MultinomialRandomUniformFusion");
    this->register_matcher(m, callback);
}
