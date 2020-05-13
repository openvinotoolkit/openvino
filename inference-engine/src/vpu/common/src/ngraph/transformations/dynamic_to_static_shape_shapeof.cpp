// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_shapeof.hpp"

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/utils/error.hpp>

#include <ngraph/opsets/opset3.hpp>

namespace vpu {

DynamicToStaticShapeShapeOf::DynamicToStaticShapeShapeOf() : GraphRewrite() {
    // We don't set strict_mode when use pattern Matcher,
    // so we can set any type and shape for input.
    auto inputWithAnyTypeAndShape = std::make_shared<ngraph::pattern::op::Label>(
            ngraph::element::dynamic, ngraph::PartialShape{});
    auto shapeOfPattern = std::make_shared<ngraph::opset3::ShapeOf>(inputWithAnyTypeAndShape);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) {
        auto shapeOfNode = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(m.get_match_root());
        if (!shapeOfNode) {
            return false;
        }

        auto dsr = shapeOfNode->input_value(0).get_node_shared_ptr();
        if (!ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dsr)) {
            return false;
        }

        ngraph::replace_node(shapeOfNode, dsr->input_value(1).get_node_shared_ptr());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shapeOfPattern, "DynamicToStaticShapeShapeOf");
    this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

} // namespace vpu
