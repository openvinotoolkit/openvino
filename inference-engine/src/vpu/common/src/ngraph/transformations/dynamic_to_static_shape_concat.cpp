// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_concat.hpp"

#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"
#include <vpu/utils/error.hpp>

#include "ngraph/graph_util.hpp"
#include "ngraph/opsets/opset3.hpp"

#include <memory>
#include <numeric>
#include <utility>

namespace vpu {

void dynamicToStaticShapeConcat(std::shared_ptr<ngraph::Node> target) {
    const auto inputs = target->input_values();

    ngraph::OutputVector dsrInputs;
    ngraph::OutputVector staticInputs;
    for (const auto& input : inputs) {
        const auto inputNode = input.get_node_shared_ptr();
        if (ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(inputNode)) {
            dsrInputs.emplace_back(input);
        } else {
            staticInputs.emplace_back(input);
        }
    }

    VPU_THROW_UNLESS(!dsrInputs.empty(),
                     "DynamicToStaticShape transformation for {} of type {} expects at least "
                     "one {} as input, actual types: {}", target->get_friendly_name(),
                     target->get_type_info(), ngraph::vpu::op::DynamicShapeResolver::type_info,
                     std::accumulate(inputs.begin(), inputs.end(), std::string(), [](
                             const std::string& typesStr, const ngraph::Output<ngraph::Node>& input) {
                         return typesStr + input.get_node_shared_ptr()->get_type_info().name + ", ";
                     }));

    const auto firstDSRInputNode = dsrInputs.front().get_node_shared_ptr();
    const auto shapeDataType = firstDSRInputNode->input(1).get_element_type();
    const auto dataRank = firstDSRInputNode->get_output_partial_shape(0).rank().get_length();
    const auto axis = ngraph::as_type_ptr<ngraph::opset3::Concat>(target)->get_concatenation_axis();

    const auto getShapeFromDSR = [&target, &shapeDataType](const ngraph::Output<ngraph::Node>& dsrOutput) {
        const auto dsrNode = dsrOutput.get_node_shared_ptr();
        const auto dsrShapeInputValue = dsrNode->input_value(1);
        VPU_THROW_UNLESS(dsrShapeInputValue.get_element_type() == shapeDataType,
                         "DynamicToStaticShape transformation for {} of type {} expects input "
                         "shape with {} type from {} argument of type {}, provided {}",
                         target->get_friendly_name(), target->get_type_info(),
                         shapeDataType, dsrNode->get_friendly_name(), dsrNode->get_type_info(),
                         dsrShapeInputValue.get_element_type());
        return dsrShapeInputValue;
    };

    const auto sumOfShapes = [](const ngraph::Output<ngraph::Node>& shape1,
                                const ngraph::Output<ngraph::Node>& shape2) {
        const auto shapeAccumulatorOp = std::make_shared<ngraph::opset3::Add>(shape1, shape2);
        return shapeAccumulatorOp->output(0);
    };

    const auto divideDimsByNumOfInputsExceptAxis = [&target, &dataRank, &axis, &shapeDataType](
            const ngraph::Output<ngraph::Node>& shape) {
        ngraph::Shape dividerValues(dataRank, target->get_input_size());
        dividerValues[axis] = 1;
        const auto divider = shapeToConstant(shapeDataType, dividerValues);
        const auto divide = std::make_shared<ngraph::opset3::Divide>(shape, divider);
        return divide->output(0);
    };

    const auto getAdditionalShapeFromStatic = [&target, &dataRank, &axis](
            const ngraph::OutputVector& staticInputs) {
        ngraph::Shape accumulatedStaticShapeValue(dataRank, 0);
        for (const auto& staticInput : staticInputs) {
            const auto& staticInputPartialShape = staticInput.get_partial_shape();
            VPU_THROW_UNLESS(staticInputPartialShape.is_static(),
                             "DynamicToStaticShape transformation for {} of type {} expects static "
                             "shape on inputs without DSR", target->get_friendly_name(),
                             target->get_type_info());
            accumulatedStaticShapeValue[axis] += staticInputPartialShape[axis].get_length();
        }
        return accumulatedStaticShapeValue;
    };

    auto accumulatedShape = getShapeFromDSR(dsrInputs.front());
    for (size_t dsrInputIdx = 1; dsrInputIdx < dsrInputs.size(); ++dsrInputIdx) {
        const auto dsrInputShape = getShapeFromDSR(dsrInputs[dsrInputIdx]);
        accumulatedShape = sumOfShapes(accumulatedShape, dsrInputShape);
    }

    if (dsrInputs.size() > 1) {
        accumulatedShape = divideDimsByNumOfInputsExceptAxis(accumulatedShape);
    }

    if (!staticInputs.empty()) {
        const auto accumulatedStaticShape = shapeToConstant(shapeDataType, getAdditionalShapeFromStatic(staticInputs));
        accumulatedShape = sumOfShapes(accumulatedShape, accumulatedStaticShape);
    }

    const auto copied = target->clone_with_new_inputs(target->input_values());

    auto outDSR = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(copied, accumulatedShape);
    outDSR->set_friendly_name(target->get_friendly_name());
    ngraph::replace_node(std::move(target), std::move(outDSR));
}

}  // namespace vpu
