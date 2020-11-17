// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/merge_subsequent_dsr_operations.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

NGRAPH_RTTI_DEFINITION(vpu::MergeSubsequentDSROperations, "MergeSubsequentDSROperations", 0);

namespace vpu {

MergeSubsequentDSROperations::MergeSubsequentDSROperations() : ngraph::pass::GraphRewrite() {
    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        const auto& dsr = std::dynamic_pointer_cast<ngraph::vpu::op::DynamicShapeResolver>(m.get_match_root());
        if (!dsr) {
            return false;
        }

        const auto& predecessor = std::dynamic_pointer_cast<ngraph::vpu::op::DynamicShapeResolver>(dsr->input_value(0).get_node_shared_ptr());
        if (!predecessor) {
            return false;
        }

        dsr->input(0).replace_source_output(predecessor->input_value(0));
        return false;
    };

    const auto& label = std::make_shared<ngraph::pattern::op::Label>(
        ngraph::element::i64,
        ngraph::Shape{},
        ngraph::pattern::has_class<ngraph::vpu::op::DynamicShapeResolver>());

    const auto& matcher = std::make_shared<ngraph::pattern::Matcher>(label, "MergeSubsequentDSROperations");
    add_matcher(matcher, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

}  // namespace vpu
