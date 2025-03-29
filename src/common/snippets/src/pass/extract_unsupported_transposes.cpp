// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/extract_unsupported_transposes.hpp"

#include "openvino/opsets/opset1.hpp"
#include "snippets/pass/mha_tokenization.hpp"
#include "snippets/itt.hpp"


bool ov::snippets::pass::ExtractUnsupportedTransposes::run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractUnsupportedTransposes");
    const auto& body = subgraph->body_ptr();
    const auto parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs,
    //           we cannot guarantee correct extraction since we don't have correct connections between body I/O and Subgraph I/O.
    OPENVINO_ASSERT(parameters.size() == subgraph->input_values().size(),
                    "Failed to extract unsupported transposes: the count of Parameters isn't equal to Subgraph inputs");

    bool updated = false;
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& parameter = parameters[i];
        const auto& consumers = parameter->get_output_target_inputs(0);
        if (consumers.size() != 1)
            continue;

        const auto transpose = ov::as_type_ptr<opset1::Transpose>(consumers.begin()->get_node()->shared_from_this());
        if (!transpose)
            continue;

        const auto& order = ov::as_type_ptr<opset1::Constant>(transpose->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(order, "ExtractUnsupportedTransposes expects Transposes with constant order");

        const auto order_value = order->cast_vector<int>();
        const auto transpose_child = *(transpose->get_output_target_inputs(0).begin());
        const auto is_brgemm_case = ov::is_type<opset1::MatMul>(transpose_child.get_node()->shared_from_this());
        // If Transpose is supported (can be decomposed or fused into Brgemm), skip
        // [116568]: It should be covered by TransposeDecomposition::is_supported or FuseTransposeBrgemm::is_supported
        if ((is_brgemm_case && TokenizeMHASnippets::get_fusion_transpose_order(order_value.size()) == order_value) ||
            (TokenizeMHASnippets::get_decomposed_transpose_order(order_value.size()) == order_value))
            continue;

        // If the transpose isn't supported - we have to extract it from Subgraph
        transpose->set_argument(0, subgraph->input_value(i));
        subgraph->set_argument(i, transpose);
        transpose_child.replace_source_output(parameter);
        parameter->set_partial_shape(transpose->get_output_partial_shape(0));
        updated = true;
    }

    if (updated) {
        subgraph->validate_and_infer_types();
    }

    return updated;
}
