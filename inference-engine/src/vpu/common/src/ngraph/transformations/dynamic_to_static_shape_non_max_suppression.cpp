// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_non_max_suppression.hpp"

#include <vpu/ngraph/operations/static_shape_non_maximum_suppression.hpp>
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"

#include <vpu/utils/error.hpp>
#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset5.hpp"

#include <memory>

namespace vpu {

void dynamicToStaticNonMaxSuppression(std::shared_ptr<ngraph::Node> node) {
    auto nms = std::dynamic_pointer_cast<ngraph::opset5::NonMaxSuppression>(node);
    VPU_THROW_UNLESS(nms, "dynamicToStaticNonMaxSuppression transformation for {} of type {} expects {} as node for replacement",
                     node->get_friendly_name(), node->get_type_info(), ngraph::opset5::NonMaxSuppression::type_info);

    auto staticShapeNMS = std::make_shared<ngraph::vpu::op::StaticShapeNonMaxSuppression>(*nms);

    auto dsrIndices = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            staticShapeNMS->output(0), staticShapeNMS->output(2));
    auto dsrScores = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            staticShapeNMS->output(1), staticShapeNMS->output(2));
    dsrIndices->set_friendly_name(nms->output(0).get_node_shared_ptr()->get_friendly_name() + ".0");
    dsrScores->set_friendly_name(nms->output(1).get_node_shared_ptr()->get_friendly_name() + ".1");

    const auto gatherValidOutputs = std::make_shared<ngraph::opset5::Gather>(
            staticShapeNMS->output(2),
            ngraph::opset5::Constant::create(staticShapeNMS->output(2).get_element_type(), ngraph::Shape{1}, {0}),
            ngraph::opset5::Constant::create(staticShapeNMS->output(2).get_element_type(), ngraph::Shape{1}, {0}));
    gatherValidOutputs->set_friendly_name(nms->output(2).get_node_shared_ptr()->get_friendly_name() + ".2");

    nms->output(0).replace(dsrIndices);
    nms->output(1).replace(dsrScores);
    nms->output(2).replace(gatherValidOutputs);
}

}  // namespace vpu
