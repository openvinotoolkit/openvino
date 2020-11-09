// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_gather.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>
#include "ngraph/graph_util.hpp"

#include "ngraph/opsets/opset5.hpp"

#include <numeric>

namespace vpu {

void dynamicToStaticShapeGatherND(std::shared_ptr<ngraph::Node> target) {
    const auto gatherND = ngraph::as_type_ptr<ngraph::opset5::GatherND>(target);
    VPU_THROW_UNLESS(gatherND, "dynamicToStaticShapeGatherND transformation is not applicable for {}, it should be {} instead",
                     target, ngraph::opset5::GatherND::type_info);

    auto shapeToConstant = [&gatherND](const ngraph::Output<ngraph::Node>& output,
                                       const ngraph::element::Type& elemType) -> std::shared_ptr<ngraph::opset5::Constant> {
        VPU_THROW_UNLESS(output.get_partial_shape().is_static(),
                         "dynamicToStaticShapeGatherND transformation for {} of type {} expects static shape on inputs without DSR",
                         gatherND->get_friendly_name(), gatherND->get_type_info());
        return ngraph::opset5::Constant::create(elemType, {output.get_shape().size()}, output.get_shape());
    };

    const auto dataDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(gatherND->input_value(0).get_node_shared_ptr());
    const auto indicesDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(gatherND->input_value(1).get_node_shared_ptr());

    VPU_THROW_UNLESS(dataDSR || indicesDSR, "dynamicToStaticShapeGatherND transformation for {} of type {} expects at least one DSR as input",
                     gatherND->get_friendly_name(), gatherND->get_type_info());
    if (dataDSR && indicesDSR) {
        VPU_THROW_UNLESS(dataDSR->get_input_element_type(1) == indicesDSR->get_input_element_type(1),
                         "dynamicToStaticShapeGatherND transformation for {} of type {} expects equal shapes data types, actual {} vs {}",
                         gatherND->get_friendly_name(), gatherND->get_type_info(),
                         dataDSR->get_input_element_type(1), indicesDSR->get_input_element_type(1));
    }
    const auto shapeElementType = indicesDSR ? indicesDSR->get_input_element_type(1) : dataDSR->get_input_element_type(1);

    const auto dataShape = dataDSR ? dataDSR->input_value(1) : shapeToConstant(gatherND->input_value(0), shapeElementType);
    const auto indicesShape = indicesDSR ? indicesDSR->input_value(1) : shapeToConstant(gatherND->input_value(1), shapeElementType);

    const auto dataShapeRank = static_cast<size_t>(dataShape.get_shape()[0]);
    const auto indicesShapeRank = static_cast<size_t>(indicesShape.get_shape()[0]);

    int64_t batchDims = gatherND->get_batch_dims();
    VPU_THROW_UNLESS(batchDims >= 0 && batchDims < std::min(dataShapeRank, indicesShapeRank),
                     "dynamicToStaticShapeGatherND: node {} has unsupported batch_dims which is expected to be"
                     " in [0; min({}, {})), but {} was provided", gatherND->get_friendly_name(), dataShapeRank, indicesShapeRank, batchDims);

    VPU_THROW_UNLESS(indicesShapeRank - batchDims >= 2,
                     "dynamicToStaticShapeGatherND: node {} should have at least two non-batch dimension, {} provided",
                     gatherND->get_friendly_name(), indicesShapeRank - batchDims);

    std::vector<int64_t> indicesShapePart(indicesShapeRank - batchDims - 1);
    std::iota(indicesShapePart.begin(), indicesShapePart.end(), batchDims);

    std::shared_ptr<ngraph::Node> outputShape = std::make_shared<ngraph::opset5::Gather>(
        indicesShape,
        ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{indicesShapePart.size()}, indicesShapePart),
        ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0}));

    const auto lastIndicesDim = gatherND->get_input_partial_shape(1)[indicesShapeRank - 1].get_length();
    if (lastIndicesDim + batchDims < dataShapeRank) {
        std::vector<int64_t> dataShapePart(dataShapeRank - batchDims - lastIndicesDim);
        std::iota(dataShapePart.begin(), dataShapePart.end(), lastIndicesDim + batchDims);
        const auto dataShapePartNode = std::make_shared<ngraph::opset5::Gather>(
                dataShape,
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{dataShapePart.size()}, dataShapePart),
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0}));
        outputShape = std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector{outputShape, dataShapePartNode}, 0);
    }

    if (batchDims > 0) {
        std::shared_ptr<ngraph::Node> batchDimsPart = std::make_shared<ngraph::opset5::Gather>(
            indicesShape,
            ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0}),
            ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0}));

        for (size_t i = 1; i < batchDims; i++) {
            const auto batchDimI = std::make_shared<ngraph::opset5::Gather>(
                indicesShape,
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {i}),
                ngraph::opset5::Constant::create(shapeElementType, ngraph::Shape{1}, {0}));

            batchDimsPart = std::make_shared<ngraph::opset5::Multiply>(batchDimsPart, batchDimI);
        }

        outputShape = std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector{batchDimsPart, outputShape}, 0);
    }

    const auto copied = target->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, outputShape);
    outDSR->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(target, std::move(outDSR));
}

}  // namespace vpu
