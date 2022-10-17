// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/common_optimizations.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "transformations/utils/utils.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::CommonOptimizations, "Snippets::CommonOptimizations", 0);

namespace ngraph {
namespace snippets {
namespace pass {


// Move up Constants which aren't scalars from body to Subgraph and replace them with Parameters inside body
void ConvertConstantsToParameters(const std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ConvertConstantsToParameters");
    auto body = subgraph->get_body();

    ParameterVector new_parameters;
    OutputVector new_external_inputs = subgraph->input_values();

    for (auto& op : body->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!(constant && ngraph::shape_size(constant->get_shape()) != 1ul))
            continue;

        auto parameter = std::make_shared<opset1::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());
        parameter->set_friendly_name(constant->get_friendly_name());
        ngraph::copy_runtime_info(constant, parameter);
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
    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::CommonOptimizations");

        auto subgraph = ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        auto body = subgraph->get_body();
        const auto is_quantized = subgraph->is_quantized();

        // Firsly we should transform all original Converts inside body to ConvertTruncation to save original behavior.
        // Then if Subgraph contains FakeQuantize we enable specific transformation for quantized subgraphs.
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::snippets::pass::TransformConvertToConvertTruncation>();
        if (is_quantized) {
            manager.register_pass<ngraph::snippets::pass::CommonFakeQuantizeDecomposition>();
        }
        manager.run_passes(body);

        // At the moment only non-scalar Constants of FakeQuantize can be inside Subgraph
        // so we can enable ConvertConstantsToParameters pass for quantized models
        if (is_quantized) {
            ConvertConstantsToParameters(subgraph);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<ngraph::snippets::op::Subgraph>(), "snippets::pass::CommonOptimizations");
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
