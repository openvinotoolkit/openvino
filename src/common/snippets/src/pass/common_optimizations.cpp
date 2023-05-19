// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/common_optimizations.hpp"

#include <memory>
#include "openvino/opsets/opset1.hpp"
#include <ngraph/pass/constant_folding.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "transformations/utils/utils.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/softmax_reshape_elimination.hpp"
#include "snippets/pass/explicit_transpose_matmul_inputs.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace pass {


// Move up Constants which aren't scalars from body to Subgraph and replace them with Parameters inside body
void ConvertConstantsToParameters(const std::shared_ptr<ov::snippets::op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ConvertConstantsToParameters");
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

CommonOptimizations::CommonOptimizations() {
    MATCHER_SCOPE(CommonOptimizations);
    ov::graph_rewrite_callback callback = [this](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::CommonOptimizations");

        auto subgraph = ov::as_type_ptr<ov::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        auto body = subgraph->body_ptr();
        const auto is_quantized = subgraph->is_quantized();

        // Firsly we should transform all original Converts inside body to ConvertTruncation to save original behavior.
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
        // so we can enable ConvertConstantsToParameters pass for quantized models
        if (is_quantized) {
            ConvertConstantsToParameters(subgraph);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::snippets::op::Subgraph>(),
                                                        matcher_name);
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ov
