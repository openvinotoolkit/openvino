// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset6.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_loop.hpp"
#include <vpu/utils/error.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <ngraph/ngraph.hpp>
#include <vpu/ngraph/operations/static_shape_loop.hpp>

template<class DataObject>
bool hasDynamic(const std::vector<DataObject>& dataObjects) {
    return std::any_of(dataObjects.cbegin(), dataObjects.cend(), [](const DataObject& data) { return data.get_partial_shape().is_dynamic(); });
}

namespace vpu {

void validateLoop(const ngraph::Node& node) {
    const auto& loop = dynamic_cast<const ngraph::opset6::Loop&>(node);
    VPU_THROW_UNLESS(loop.get_input_size() >= 3, "Encountered operation {} with {} inputs, expected at least {} inputs", loop, loop.get_input_size(), 3);

    const auto& executionCondition = ngraph::as_type_ptr<ngraph::opset6::Constant>(loop.input_value(1).get_node_shared_ptr());
    VPU_THROW_UNLESS(executionCondition != nullptr, "Execution condition of a loop {} is expected to be constant true, got {}", loop, executionCondition);
    const auto& executionConditionValue = executionCondition->get_vector<bool>();
    VPU_THROW_UNLESS(executionConditionValue == std::vector<bool>{true},
        "Execution condition of a loop {} is expected to be constant true, got {}", loop, executionCondition);

    for (const auto& inputDescription : loop.get_input_descriptions()) {
        if (const auto& sliceInputDescription = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::SliceInputDescription>(inputDescription)) {
            const auto& sliceInput = loop.input_value(sliceInputDescription->m_input_index);
            const auto& partialShape = sliceInput.get_partial_shape();

            VPU_THROW_UNLESS(partialShape.rank().is_static(), "Slice input {} of a loop {} is expected to have static rank, got dynamic", sliceInput, loop);
            const auto& rank = partialShape.rank().get_length();
            for (std::size_t dimension = 1; dimension < rank; ++dimension) {
                VPU_THROW_UNLESS(partialShape[dimension].is_static(),
                    "Slice input {} of a loop {} is expected to have only batch as dynamic dimension, got {}", sliceInput, loop, partialShape);
            }
        } else if (const auto& invariantInputDescription = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::InvariantInputDescription>(inputDescription)) {
            const auto& invariantInput = loop.input_value(invariantInputDescription->m_input_index);
            const auto& partialShape = invariantInput.get_partial_shape();
            VPU_THROW_UNLESS(partialShape.is_static(),
                "Invariant input {} of a loop {} is expected to have static shape, got {}", invariantInput, loop, partialShape);
        } else {
            VPU_THROW_FORMAT("Encountered unknown input type of a loop {} at index {}", loop, inputDescription->m_input_index);
        }
    }

    const auto& body = loop.get_function();
    for (const auto& operation : body->get_ordered_ops()) {
        VPU_THROW_UNLESS(!hasDynamic(operation->inputs()) && !hasDynamic(operation->outputs()),
            "Encountered a loop {} with dynamic operation {} in the body, but only static body loops are supported", loop, operation);
    }

    for (const auto& outputDescription : loop.get_output_descriptions()) {
        if (const auto& concatOutputDescription = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(outputDescription)) {
            const auto& concatOutput = loop.output(concatOutputDescription->m_output_index);
            const auto& partialShape = concatOutput.get_partial_shape();

            VPU_THROW_UNLESS(partialShape.rank().is_static(), "Concat output {} of a loop {} is expected to have static rank, got dynamic", concatOutput, loop);
            const auto& rank = partialShape.rank().get_length();
            for (std::size_t dimension = 1; dimension < rank; ++dimension) {
                VPU_THROW_UNLESS(partialShape[dimension].is_static(),
                                 "Concat output {} of a loop {} is expected to have only batch as dynamic dimension, got {}", concatOutput, loop, partialShape);
            }
        } else if (const auto& bodyOutputDescription = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::BodyOutputDescription>(outputDescription)) {
            const auto& bodyOutput = loop.output(bodyOutputDescription->m_output_index);
            const auto& partialShape = bodyOutput.get_partial_shape();
            VPU_THROW_UNLESS(partialShape.is_static(),
                             "Body output {} of a loop {} is expected to have static shape, got {}", bodyOutput, loop, partialShape);
        } else {
            VPU_THROW_FORMAT("Encountered unknown output type of a loop {} at index {}", loop, outputDescription->m_output_index);
        }
    }
}

void dynamicToStaticShapeLoop(std::shared_ptr<ngraph::Node> node) {
    const auto& loop = ngraph::as_type_ptr<ngraph::opset6::Loop>(node);
    VPU_THROW_UNLESS(loop != nullptr, "Encountered node {}, but expected loop", node);

    const auto copied = ngraph::as_type_ptr<ngraph::opset6::Loop>(loop->clone_with_new_inputs(loop->input_values()));
    const auto& staticShapeLoop = std::make_shared<ngraph::vpu::op::StaticShapeLoop>(*copied);
    staticShapeLoop->validate_and_infer_types();
    const auto& iterationsCount = staticShapeLoop->input_value(0);
    const auto& body = staticShapeLoop->get_function();
    for (const auto& outputDescription : loop->get_output_descriptions()) {
        const auto& index = outputDescription->m_output_index;
        auto replacement = staticShapeLoop->output(index).get_node_shared_ptr();
        if (const auto& concatOutputDescription = ngraph::as_type_ptr<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(outputDescription)) {
            const auto& bodyOutput = body->get_results().at(concatOutputDescription->m_body_value_index)->input_value(0);

            VPU_THROW_UNLESS(bodyOutput.get_partial_shape().is_static(),
                             "Encountered loop {} with dynamic body output {}, but only static body is supported", loop, bodyOutput);
            auto shape = bodyOutput.get_shape();
            const auto& axis = concatOutputDescription->m_axis;

            const auto outputShape = std::make_shared<ngraph::opset6::ScatterElementsUpdate>(
                std::make_shared<ngraph::opset6::Constant>(ngraph::element::i64, ngraph::Shape{shape.size()}, shape),
                std::make_shared<ngraph::opset6::Constant>(ngraph::element::i64, ngraph::Shape{1}, axis),
                iterationsCount,
                std::make_shared<ngraph::opset6::Constant>(ngraph::element::i64, ngraph::Shape{}, 0));

            replacement = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(replacement, outputShape);
        }

        replacement->set_friendly_name(loop->get_friendly_name() + "." + std::to_string(index));
        loop->output(index).replace(replacement);
    }
}

}  // namespace vpu
