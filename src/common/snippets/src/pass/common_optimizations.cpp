// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/common_optimizations.hpp"

#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/softmax_reshape_elimination.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {


void CommonOptimizations::ExtractConstants(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractConstants");
    auto body = subgraph->body_ptr();

    ParameterVector new_parameters;
    OutputVector new_external_inputs = subgraph->input_values();

    for (auto& op : body->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || ov::shape_size(constant->get_shape()) == 1ul)
            continue;

        const auto child = constant->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        if (op::Subgraph::constant_input_should_be_inside_body(child))
            continue;

        auto parameter = std::make_shared<ov::op::v0::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());
        parameter->set_friendly_name(constant->get_friendly_name());
        ov::copy_runtime_info(constant, parameter);
        constant->output(0).replace(parameter->output(0));

        new_external_inputs.push_back(constant);
        new_parameters.push_back(parameter);
    }

    if (new_parameters.size() != 0) {
        body->add_parameters(new_parameters);
        body->validate_nodes_and_infer_types();
        subgraph->set_arguments(new_external_inputs);
    }
}

void CommonOptimizations::ExtractUnsupportedTransposes(const std::shared_ptr<op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractUnsupportedTransposes");
    const auto& body = subgraph->body_ptr();
    const auto parameters = body->get_parameters();
    // [107806]: If count of Parameters isn't equal to Subgraph inputs,
    //           we cannot guarantee correct extraction since we don't have correct connections between body I/O and Subgraph I/O.
    if (parameters.size() != subgraph->input_values().size())
        return;

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
        if (!order)
            continue;

        const auto order_value = order->cast_vector<int>();
        const auto transpose_child = *(transpose->get_output_target_inputs(0).begin());
        const auto is_brgemm_case = ov::is_type<opset1::MatMul>(transpose_child.get_node()->shared_from_this());
        // If Transpose is supported (can be decomposed or fused into Brgemm), skip
        if ((is_brgemm_case && FuseTransposeBrgemm::supported_cases.count(order_value) != 0) ||
            (TransposeDecomposition::supported_cases.count(order_value) != 0))
            continue;

        // If the transpose isn't supported - we have to extract it from Subgraph
        transpose->set_argument(0, subgraph->input_value(i));
        subgraph->set_argument(i, transpose);
        transpose_child.replace_source_output(parameter);
        // Update shape
        parameter->set_partial_shape(transpose->get_output_partial_shape(0));
        updated = true;
    }

    if (updated) {
        subgraph->validate_and_infer_types();
    }
}

CommonOptimizations::CommonOptimizations() {
    MATCHER_SCOPE(CommonOptimizations);
    ov::graph_rewrite_callback callback = [this](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::CommonOptimizations");

        auto subgraph = ov::as_type_ptr<ov::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        const auto& body = subgraph->body_ptr();
        const auto is_quantized = subgraph->is_quantized();

        // Firstly, we should transform all original Converts inside body to ConvertTruncation to save original behavior.
        // Then if Subgraph contains FakeQuantize we enable specific transformation for quantized subgraphs.
        ov::pass::Manager manager;
        manager.register_pass<ov::snippets::pass::TransformConvertToConvertTruncation>();
        manager.register_pass<ov::snippets::pass::ExplicitTransposeMatMulInputs>();
        if (is_quantized) {
            manager.register_pass<ov::snippets::pass::CommonFakeQuantizeDecomposition>();
        }
        manager.register_pass<snippets::pass::SoftmaxReshapeElimination>();
        manager.run_passes(body);

        // At the moment only non-scalar Constants of FakeQuantize can be inside Subgraph
        // so we can enable ExtractConstants pass for quantized models
        if (is_quantized) {
            ExtractConstants(subgraph);
        }
        // Extract unsupported Transposes from body
        if (subgraph->has_domain_sensitive_ops()) {
            ExtractUnsupportedTransposes(subgraph);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::snippets::op::Subgraph>(), matcher_name);
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ov
