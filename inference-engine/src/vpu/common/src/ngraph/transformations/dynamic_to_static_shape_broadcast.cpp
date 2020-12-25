// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_broadcast.hpp"

#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include "vpu/utils/error.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset5.hpp"

#include <memory>
#include <algorithm>

namespace vpu {

void dynamicToStaticShapeBroadcast(std::shared_ptr<ngraph::Node> target) {
    const auto broadcast = ngraph::as_type_ptr<ngraph::opset3::Broadcast>(target);
    VPU_THROW_UNLESS(broadcast,
                     "dynamicToStaticShapeBroadcast transformation is not applicable for {}, "
                     "it should be {} instead",
                     target, ngraph::opset3::Broadcast::type_info);

    std::shared_ptr<ngraph::vpu::op::StaticShapeBroadcast> staticShapeBroadcast;
    if (broadcast->get_broadcast_spec() == ngraph::op::BroadcastType::EXPLICIT) {
        staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                broadcast->input_value(0),
                broadcast->input_value(1),
                broadcast->input_value(2));
    } else if (broadcast->get_broadcast_spec() == ngraph::op::BroadcastType::NUMPY ||
               broadcast->get_broadcast_spec() == ngraph::op::BroadcastType::BIDIRECTIONAL) {
        staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
                broadcast->input_value(0),
                broadcast->input_value(1),
                broadcast->get_broadcast_spec());
    } else {
        VPU_THROW_FORMAT("dynamicToStaticShapeBroadcast supports only explicit, numpy and bidirectional modes,"
                         "provided {}", broadcast->get_broadcast_spec().m_type);
    }

    std::shared_ptr<ngraph::Node> dsr;

    if (broadcast->get_broadcast_spec() == ngraph::op::BroadcastType::BIDIRECTIONAL) {
        const auto dataDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(broadcast->input_value(0).get_node_shared_ptr());
        const auto shapeElementType = dataDSR ? dataDSR->get_input_element_type(1) : broadcast->get_input_element_type(1);
        const auto dataShape = dataDSR ? dataDSR->input_value(1) : shapeToConstant(shapeElementType, broadcast->get_input_shape(0));

        const auto targetShape = broadcast->input_value(1).get_node_shared_ptr();

        const auto dataShapeDimsCount = ngraph::shape_size(dataShape.get_shape());
        const auto targetShapeDimsCount = ngraph::shape_size(broadcast->get_input_partial_shape(1).get_shape());

        const auto minRank = std::min(dataShapeDimsCount, targetShapeDimsCount);
        const auto maxRank = std::max(dataShapeDimsCount, targetShapeDimsCount);
        const auto minRankNode = minRank == dataShapeDimsCount ? dataShape : targetShape;
        const auto maxRankNode = minRank == dataShapeDimsCount ? targetShape : dataShape;

        ngraph::NodeVector dims;

        for (int i = 0; i < maxRank - minRank; i++) {
            dims.push_back(
                std::make_shared<ngraph::opset5::Gather>(
                    maxRankNode,
                    ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {i}),
                    ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0})));
        }

        for (int i = 0; i < minRank; i++) {
            const auto minRankDim = std::make_shared<ngraph::opset5::Gather>(
                minRankNode,
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {i}),
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0}));
            const auto maxRankDim = std::make_shared<ngraph::opset5::Gather>(
                maxRankNode,
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {maxRank - minRank + i}),
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0}));
            dims.push_back(std::make_shared<ngraph::opset5::Maximum>(minRankDim, maxRankDim));
        }

        const auto outShape = std::make_shared<ngraph::opset5::Concat>(dims, 0);

        dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(staticShapeBroadcast->output(0), outShape);
    } else {
        dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(staticShapeBroadcast->output(0), broadcast->input_value(1));
    }

    dsr->set_friendly_name(broadcast->get_friendly_name());
    ngraph::replace_node(std::move(target), std::move(dsr));
}

}  // namespace vpu

