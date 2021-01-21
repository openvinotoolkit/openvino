// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/eliminate_shapeof_after_dsr.hpp"

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/utils/error.hpp>

#include <ngraph/opsets/opset3.hpp>

NGRAPH_RTTI_DEFINITION(vpu::EliminateShapeOfAfterDSR, "EliminateShapeOfAfterDSR", 0);

namespace vpu {

EliminateShapeOfAfterDSR::EliminateShapeOfAfterDSR() {
    // We don't set strict_mode when use pattern Matcher,
    // so we can set any type and shape for input.
    auto inputWithAnyTypeAndShape = std::make_shared<ngraph::pattern::op::Label>(
            ngraph::element::dynamic, ngraph::PartialShape{});
    auto shapeOfPattern = std::make_shared<ngraph::opset3::ShapeOf>(inputWithAnyTypeAndShape);

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto shapeOfNode = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(m.get_match_root());
        if (!shapeOfNode) {
            return false;
        }

        auto dsr = shapeOfNode->input_value(0).get_node_shared_ptr();
        if (!ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr)) {
            return false;
        }

        shapeOfNode->output(0).replace(dsr->input_value(1));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shapeOfPattern, "EliminateShapeOfAfterDSR");
    register_matcher(m, callback);
}

} // namespace vpu
