// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/align_element_types.hpp"
#include "snippets/itt.hpp"

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
            std::shared_ptr<ov::Node> consumer = results[i];
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

            auto parent_output = parameter->output(0);
            auto consumer_inputs = parent_output.get_target_inputs();

            const auto& first_child = consumer_inputs.begin()->get_node()->shared_from_this();
            // Note: RankNormalization of is designed for shape-inference purposes only.
            // It does not process any data (nor does it emit any code), so it doesn't require Convert operations
            if (is_type<op::RankNormalization>(first_child)) {
                OPENVINO_ASSERT(consumer_inputs.size() == 1, "RankNormalization is supposed to be the only consumer");
                parent_output = first_child->output(0);
                consumer_inputs = parent_output.get_target_inputs();
            }

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