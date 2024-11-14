// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/align_element_types.hpp"

#include "snippets/pass/propagate_precision.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {

pass::AlignElementTypes::AlignElementTypes(std::vector<ov::element::Type> input_precisions,
                                           std::vector<ov::element::Type> output_precisions) :
                                           m_input_precisions(std::move(input_precisions)),
                                           m_output_precisions(std::move(output_precisions)) {
}

bool pass::AlignElementTypes::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(AlignElementTypes);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::AlignElementTypes")
    bool is_modified = false;
    const auto& results = m->get_results();
    const auto& params = m->get_parameters();
    OPENVINO_ASSERT(m_input_precisions.size() == params.size() && m_output_precisions.size() == results.size(),
                    "Number of parameters for snippet doesn't match passed to the Canonicalization pass. ");

    // We should insert Convert before Results to set original output element type if needed
    for (size_t i = 0; i < m_output_precisions.size(); i++) {
        const auto needed_out_type = m_output_precisions[i];
        if (results[i]->get_input_element_type(0) != needed_out_type) {
            const auto& shape_infer_leaf = utils::get_leaf_node_of_first_parent_shape_infer_seq(results[i]);
            std::shared_ptr<ov::Node> consumer = shape_infer_leaf ? shape_infer_leaf : results[i];
            auto parent_output = consumer->get_input_source_output(0);

            // Snippets supports Transpose only after Parameter or before Result nodes
            // So we have to insert Convert before Transpose (if there is) on Subgraph outputs
            const auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(parent_output.get_node_shared_ptr());
            if (transpose) {
                OPENVINO_ASSERT(parent_output.get_target_inputs().size() == 1,
                                "If Result has Transpose on input, this Result must be single consumer of the Transpose");
                parent_output = transpose->get_input_source_output(0);
                consumer = transpose;
            }

            // If there is already Convert[needed_in_type->original_type] and this node has only one consumer, we can remove the Convert,
            // since the sequence existing Convert[needed_in_type->original_type] -> new Convert[original_type->needed_in_type] is redundant
            if (const auto existing_convert = ov::as_type_ptr<ov::snippets::op::ConvertSaturation>(parent_output.get_node_shared_ptr())) {
                const auto actual_before = existing_convert->get_input_element_type(0);
                const auto actual_after = existing_convert->get_output_element_type(0);
                const auto required_after = needed_out_type;
                if (ov::snippets::pass::PropagatePrecision::can_be_removed(actual_before, actual_after, required_after) &&
                    parent_output.get_target_inputs().size() == 1) {
                    // remove existing convert
                    existing_convert->output(0).replace(existing_convert->input_value(0));
                    continue;
                }
            }

            const auto convert = std::make_shared<op::ConvertSaturation>(parent_output, needed_out_type);
            ov::copy_runtime_info(parent_output.get_node_shared_ptr(), convert);

            consumer->set_argument(0, convert);
            consumer->validate_and_infer_types();
            if (transpose)
                results[i]->validate_and_infer_types();
            is_modified = true;
        }
    }

    // We should change existing element type to original for Parameters if needed
    for (size_t i = 0; i < m_input_precisions.size(); ++i) {
        const auto needed_in_type = m_input_precisions[i];
        const auto& parameter = params[i];
        const auto original_type = parameter->get_element_type();
        if (original_type != needed_in_type) {
            parameter->set_element_type(needed_in_type);
            parameter->validate_and_infer_types();

            // Note: shape infer ops are designed for shape-inference purposes only.
            // It does not process any data (nor does it emit any code), so it doesn't require Convert operations
            const auto& shape_infer_leaf = utils::get_leaf_node_of_first_child_shape_infer_seq(parameter);
            const auto& first_child = shape_infer_leaf ? shape_infer_leaf : parameter;
            auto parent_output = first_child->output(0);
            auto consumer_inputs = parent_output.get_target_inputs();

            // Snippets supports Transpose only after Parameter or before Result nodes
            // So we have to insert Convert after Transpose (if there is) on Subgraph inputs
            if (std::any_of(consumer_inputs.cbegin(), consumer_inputs.cend(),
                            [](const ov::Input<ov::Node>& input) { return ov::is_type<ov::op::v1::Transpose>(input.get_node()); })) {
                OPENVINO_ASSERT(consumer_inputs.size() == 1,
                                "If Parameter has Transpose on output, this Transpose must be single consumer of the Parameter");
                const auto transpose = consumer_inputs.begin()->get_node()->shared_from_this();
                transpose->validate_and_infer_types();

                parent_output = transpose;
                consumer_inputs = parent_output.get_target_inputs();
            }

            // If there is already Convert[original_type->needed_in_type] and this node is alone consumer, we can remove the Convert,
            // since the sequence new Convert[needed_in_type->original_type] -> existing Convert[original_type->needed_in_type] is redundant
            if (const auto existing_convert = ov::as_type_ptr<ov::snippets::op::ConvertSaturation>(consumer_inputs.cbegin()->get_node()->shared_from_this())) {
                const auto actual_before = needed_in_type;
                const auto actual_after = original_type;
                const auto required_after = existing_convert->get_element_type();
                if (ov::snippets::pass::PropagatePrecision::can_be_removed(actual_before, actual_after, required_after) &&
                    consumer_inputs.size() == 1) {
                    // remove existing convert
                    existing_convert->output(0).replace(parent_output);
                    continue;
                }
            }

            const auto& convert = std::make_shared<ov::snippets::op::ConvertSaturation>(parent_output, original_type);
            ov::copy_runtime_info(parent_output.get_node_shared_ptr(), convert);

            for (const auto input : consumer_inputs) {
                const auto& input_node = input.get_node();
                if (input_node == convert.get()) {
                    continue;
                }
                input_node->set_argument(input.get_index(), convert->output(0));
            }

            is_modified = true;
        }
    }
    return is_modified;
}

} // namespace snippets
} // namespace ov