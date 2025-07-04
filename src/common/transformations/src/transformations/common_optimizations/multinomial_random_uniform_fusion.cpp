// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/multinomial_random_uniform_fusion.hpp"

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

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
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
            multinomial->input_value(1),
            ov::op::v0::Constant::create(multinomial->get_input_element_type(0), ov::Shape{}, {0}),
            ov::op::v0::Constant::create(multinomial->get_input_element_type(0), ov::Shape{}, {1}),
            multinomial->get_input_element_type(0),
            multinomial->get_global_seed(),
            multinomial->get_op_seed(),
            multinomial->get_alignment());

        auto new_multinomial = std::make_shared<ov::op::v13::Multinomial>(multinomial->input_value(0),
                                                                          multinomial->input_value(1),
                                                                          random_samples,
                                                                          multinomial->get_convert_type(),
                                                                          multinomial->get_with_replacement(),
                                                                          multinomial->get_log_probs(),
                                                                          multinomial->get_global_seed(),
                                                                          multinomial->get_op_seed(),
                                                                          multinomial->get_alignment());

        new_multinomial->set_friendly_name(multinomial->get_friendly_name());
        ov::copy_runtime_info(multinomial, {random_samples, new_multinomial});
        ov::replace_node(multinomial, new_multinomial);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(multinomial_pattern, "MultinomialRandomUniformFusion");
    this->register_matcher(m, callback);
}
