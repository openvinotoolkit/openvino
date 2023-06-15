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
            const auto convert = std::make_shared<op::ConvertSaturation>(
                    results[i]->get_input_node_shared_ptr(0), needed_out_type);
            results[i]->set_argument(0, convert);
            results[i]->validate_and_infer_types();
            is_modified = true;
        }
    }

    // We should change existing element type to original for Parameters if needed
    for (size_t i = 0; i < m_input_precisions.size(); ++i) {
        const auto needed_in_type = m_input_precisions[i];
        const auto& parameter = params[i];
        if (parameter->get_element_type() != needed_in_type) {
            auto parameter_output = parameter->output(0);
            const auto& targets = parameter_output.get_target_inputs();
            const auto& first_child = targets.begin()->get_node()->shared_from_this();
            // Note: RankNormalization of is designed for shape-inference purposes only.
            // It does not process any data (nor does it emit any code), so it doesn't require Convert operations
            if (targets.size() == 1 && ov::is_type<op::RankNormalization>(first_child))
                parameter_output = first_child->output(0);

            const auto convert = std::make_shared<op::ConvertSaturation>(
                    parameter_output,
                    parameter_output.get_element_type());
            ov::copy_runtime_info(parameter, convert);

            for (const auto input : parameter_output.get_target_inputs()) {
                const auto& input_node = input.get_node();
                if (input_node == convert.get()) {
                    continue;
                }
                input_node->set_argument(input.get_index(), convert->output(0));
            }

            parameter->set_element_type(needed_in_type);
            parameter->validate_and_infer_types();
            is_modified = true;
        }
    }
    return is_modified;
}

} // namespace snippets
} // namespace ov