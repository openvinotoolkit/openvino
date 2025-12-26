// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/extract_unsupported_transposes.hpp"

#include <cstddef>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/subgraph.hpp"

bool ov::snippets::pass::ExtractUnsupportedTransposes::run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractUnsupportedTransposes");
    const auto& body = subgraph->body_ptr();
    const auto parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs,
    //           we cannot guarantee correct extraction since we don't have correct connections between body I/O and
    //           Subgraph I/O.
    OPENVINO_ASSERT(parameters.size() == subgraph->input_values().size(),
                    "Failed to extract unsupported transposes: the count of Parameters isn't equal to Subgraph inputs");

    bool updated = false;
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& parameter = parameters[i];
        const auto& consumers = parameter->get_output_target_inputs(0);
        if (consumers.size() != 1) {
            continue;
        }

        const auto transpose = ov::as_type_ptr<opset1::Transpose>(consumers.begin()->get_node()->shared_from_this());
        if (!transpose) {
            continue;
        }

        OPENVINO_ASSERT(m_transpose_support_cb,
                        "Transpose support callback is not set in ExtractUnsupportedTransposes pass");
        bool is_supported = m_transpose_support_cb(transpose);
        if (is_supported) {
            continue;
        }

        // If the transpose isn't supported - we have to extract it from Subgraph.
        const auto new_transpose =
            transpose->clone_with_new_inputs({subgraph->input_value(i), transpose->input_value(1)});
        subgraph->set_argument(i, new_transpose);
        OPENVINO_ASSERT(!transpose->get_output_target_inputs(0).empty(),
                        "ExtractUnsupportedTransposes pass supports only Transpose nodes with at least one consumer");
        const auto transpose_child = *(transpose->get_output_target_inputs(0).begin());
        transpose_child.replace_source_output(parameter);
        parameter->set_partial_shape(transpose->get_output_partial_shape(0));
        updated = true;
    }

    if (updated) {
        subgraph->validate_and_infer_types();
    }

    return updated;
}
