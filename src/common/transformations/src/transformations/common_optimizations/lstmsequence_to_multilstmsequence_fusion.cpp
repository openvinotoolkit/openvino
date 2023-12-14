// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/lstmsequence_to_multilstmsequence_fusion.hpp"

ov::pass::LSTMSequenceToMultiLSTMSequenceFusion::LSTMSequenceToMultiLSTMSequenceFusion() {
    MATCHER_SCOPE(LSTMSequenceToMultiLSTMSequenceFusion);

    // Create parallel LSTMSequences pattern
    //auto m_data = pass::pattern::any_input();
    //auto m_add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_concat = ov::pass::pattern::wrap_type<ov::op::v0::Concat>({m_data, m_add_constant}, pattern::consumers_count(1));
    auto m_mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_add, m_mul_constant});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) -> bool {
        auto& label_to_output = m.get_pattern_value_map();

        auto mul = label_to_output[m_mul].get_node_shared_ptr();
        auto add = label_to_output[m_add].get_node_shared_ptr();

        if (transformation_callback(mul)) {
            return false;
        }

        Output<Node> input = label_to_output[m_data];
        Output<Node> mul_const = label_to_output[m_mul_constant];
        Output<Node> add_const = label_to_output[m_add_constant];

        if ((input.get_element_type() != mul_const.get_element_type()) ||
            (add_const.get_element_type() != mul_const.get_element_type())) {
            return false;
        }

        // Replace Add->Multiply with Multiply->Add
        // As new Multiply can be fused with operation above it we add this Multiply
        // to the list of operations that will be used in additional matching.
        auto new_mul = register_new_node<ov::op::v1::Multiply>(input, mul_const);

        // Add two constants using opset3::Add constant folding and create new Add operation
        auto new_add =
            std::make_shared<ov::op::v1::Add>(new_mul,
                                              op::util::eltwise_fold<ov::op::v1::Multiply>(add_const, mul_const));

        copy_runtime_info({add, mul}, {new_mul, new_add});
        new_add->set_friendly_name(mul->get_friendly_name());
        replace_node(mul, new_add);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_mul, matcher_name);
    this->register_matcher(m, callback);
}