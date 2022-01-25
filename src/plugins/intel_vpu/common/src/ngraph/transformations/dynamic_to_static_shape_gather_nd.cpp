// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_gather_nd.hpp"

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
                     target, ngraph::opset5::GatherND::get_type_info_static());

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

    const auto dataShape = dataDSR ? dataDSR->input_value(1) : shapeToConstant(shapeElementType, gatherND->get_input_shape(0));
    const auto indicesShape = indicesDSR ? indicesDSR->input_value(1) : shapeToConstant(shapeElementType, gatherND->get_input_shape(1));

    const auto dataShapeRank = ngraph::shape_size(dataShape.get_shape());
    const auto indicesShapeRank = ngraph::shape_size(indicesShape.get_shape());

    const auto batchDims = static_cast<int64_t>(gatherND->get_batch_dims());
    VPU_THROW_UNLESS(batchDims >= 0 && batchDims < std::min(dataShapeRank, indicesShapeRank),
                     "dynamicToStaticShapeGatherND: node {} has unsupported batch_dims which is expected to be"
                     " in [0; min({}, {})), but {} was provided", gatherND->get_friendly_name(), dataShapeRank, indicesShapeRank, batchDims);

    std::shared_ptr<ngraph::Node> outputShape;

    if (batchDims > 0) {
        outputShape = std::make_shared<ngraph::opset5::ReduceProd>(
            gatherShapeElements(indicesShape, 0, batchDims),
            ngraph::opset5::Constant::create(ngraph::element::i64, {}, {0}),
            true);
    }

    if (indicesShapeRank - batchDims - 1 > 0) {
        const auto indicesShapePart = gatherShapeElements(indicesShape, batchDims, indicesShapeRank - batchDims - 1);
        outputShape = outputShape ? std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector{outputShape, indicesShapePart}, 0) : indicesShapePart;
    }

    const auto lastIndicesDim = gatherND->get_input_partial_shape(1)[indicesShapeRank - 1].get_length();
    if (batchDims + lastIndicesDim < dataShapeRank) {
        const auto dataShapePart = gatherShapeElements(
            dataShape,
            lastIndicesDim + batchDims,
            dataShapeRank - batchDims - lastIndicesDim);
        outputShape = outputShape ? std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector{outputShape, dataShapePart}, 0) : dataShapePart;
    }

    VPU_THROW_UNLESS(outputShape, "dynamicToStaticShapeGatherND: node {} has empty output shape", gatherND->get_friendly_name());

    const auto copied = target->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, outputShape);
    outDSR->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(target, std::move(outDSR));
}

}  // namespace vpu
