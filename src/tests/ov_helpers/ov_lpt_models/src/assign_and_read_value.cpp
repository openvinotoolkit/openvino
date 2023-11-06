// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset6.hpp>
#include "ov_models/subgraph_builders.hpp"
#include "openvino/op/util/variable.hpp"
#include <openvino/op/util/assign_base.hpp>

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/assign_and_read_value.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AssignAndReadValueFunction::getOriginal(
        const ngraph::PartialShape& inputShape,
        const element::Type& inputPrecision,
        const ngraph::element::Type precisionBeforeDequantization,
        const size_t opsetVersion,
        const bool FQAfterReadValue,
        const std::vector<float>& constantValue,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    const auto defaultConstant = std::make_shared<opset1::Constant>(inputPrecision, inputShape.get_shape(), constantValue);
    const auto variable = std::make_shared<Variable>(VariableInfo{inputShape.get_shape(), inputPrecision, "id"});
    std::shared_ptr<Node> readValue;
    if (opsetVersion == 6) {
        readValue = std::make_shared<opset6::ReadValue>(defaultConstant, variable);
    } else if (opsetVersion == 3) {
        readValue = std::make_shared<opset3::ReadValue>(defaultConstant, "id");
    } else {
        throw std::runtime_error("Unknown opset version");
    }
    std::shared_ptr<Node> lastNode = readValue;
    if (FQAfterReadValue) {
        lastNode = builder::subgraph::makeFakeQuantize(
                lastNode,
                element::f32,
                FakeQuantizeOnData{256ul, Shape{}, {0}, {2.55f}, {0}, {2.55f}});
    }
    const auto add = std::make_shared<opset1::Add>(lastNode, input);
    const auto FQAfterAdd = builder::subgraph::makeFakeQuantizeTypeRelaxed(
            add,
            element::f32,
            FakeQuantizeOnData{256ul, Shape{}, {0}, {2.55f}, {0}, {2.55f}, precisionBeforeDequantization});
    auto deqStructure = dequantization;
    deqStructure.multiply.outPrecision = inputPrecision;
    const auto dequantizationOp = makeDequantization(FQAfterAdd, deqStructure);
    std::shared_ptr<Node> assign;
    if (opsetVersion == 6) {
        assign = std::make_shared<opset6::Assign>(dequantizationOp, variable);
    } else {
        assign = std::make_shared<opset3::Assign>(dequantizationOp, "id");
    }
    assign->add_control_dependency(readValue);
    add->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    ngraph::SinkVector sinks{ as_type_ptr<ov::op::Sink>(assign) };
    return std::make_shared<ngraph::Function>(results, sinks, ngraph::ParameterVector{ input }, "AssignAndReadValueFunction");
}

std::shared_ptr<ngraph::Function> AssignAndReadValueFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const size_t opsetVersion) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
    const auto defaultConstant = std::make_shared<opset1::Constant>(precision, inputShape.get_shape(), std::vector<float>{0});
    const auto variable = std::make_shared<Variable>(VariableInfo{inputShape.get_shape(), precision, "id"});
    std::shared_ptr<Node> readValue;
    if (opsetVersion == 6) {
        readValue = std::make_shared<opset6::ReadValue>(defaultConstant, variable);
    } else if (opsetVersion == 3) {
        readValue = std::make_shared<opset3::ReadValue>(defaultConstant, "id");
    } else {
        throw std::runtime_error("Unknown opset version");
    }
    std::shared_ptr<Node> lastNode = readValue;
    lastNode = builder::subgraph::makeFakeQuantize(
            lastNode,
            element::f32,
            FakeQuantizeOnData{256ul, Shape{}, {0}, {2.55f}, {0}, {2.55f}});
    const auto add = std::make_shared<opset1::Add>(lastNode, input);
    const auto FQAfterAdd = fakeQuantize.empty() ? nullptr :
                              ngraph::builder::makeFakeQuantize(
                                      add,
                                      precision,
                                      fakeQuantize.quantizationLevel,
                                      fakeQuantize.constantShape,
                                      fakeQuantize.inputLowValues,
                                      fakeQuantize.inputHighValues,
                                      fakeQuantize.outputLowValues,
                                      fakeQuantize.outputHighValues);
    std::shared_ptr<Node> assign;
    if (opsetVersion == 6) {
        assign = std::make_shared<opset6::Assign>(FQAfterAdd, variable);
    } else {
        assign = std::make_shared<opset3::Assign>(FQAfterAdd, "id");
    }
    assign->add_control_dependency(readValue);
    add->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    ngraph::SinkVector sinks{ as_type_ptr<ov::op::Sink>(assign) };
    return std::make_shared<ngraph::Function>(results, sinks, ngraph::ParameterVector{ input }, "AssignAndReadValueFunction");
}

std::shared_ptr<ngraph::Function> AssignAndReadValueFunction::getReference(
    const ngraph::PartialShape& inputShape,
    const element::Type& inputPrecision,
    const ngraph::element::Type precisionBeforeDequantization,
    const size_t opsetVersion,
    const bool FQAfterReadValue,
    const std::vector<float>& constantValue,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    auto constantPrecision = precisionBeforeDequantization;
    if (constantValue != std::vector<float>{0}) {
        constantPrecision = inputPrecision;
    }
    const auto defaultConstant = std::make_shared<opset1::Constant>(constantPrecision, inputShape.get_shape(), constantValue);
    const auto variable = std::make_shared<Variable>(VariableInfo{inputShape.get_shape(), constantPrecision, "id"});
    std::shared_ptr<Node> readValue;
    if (opsetVersion == 6) {
        readValue = std::make_shared<opset6::ReadValue>(defaultConstant, variable);
    } else if (opsetVersion == 3) {
        readValue = std::make_shared<opset3::ReadValue>(defaultConstant, "id");
    } else {
        throw std::runtime_error("Unknown opset version");
    }
    std::shared_ptr<Node> lastNode = readValue;

    auto deqStructureAfter = dequantizationAfter;
    if (FQAfterReadValue) {
        DequantizationOperations tempDequantization;
        tempDequantization.convert = dequantizationAfter.convert;
        tempDequantization.subtract = dequantizationAfter.subtract;
        lastNode = makeDequantization(lastNode, tempDequantization);
    } else {
        deqStructureAfter.multiply.outPrecision = inputPrecision;
        lastNode = makeDequantization(lastNode, deqStructureAfter);
    }

    if (FQAfterReadValue) {
        lastNode = builder::subgraph::makeFakeQuantizeTypeRelaxed(
                lastNode,
                element::f32,
                FakeQuantizeOnData{256ul, Shape{}, {0}, {2.55f / dequantizationAfter.multiply.values[0]}, {0}, {2.55f}, inputPrecision});
    }
    const auto add = std::make_shared<opset1::Add>(lastNode, input);
    const auto FQAfterAdd = builder::subgraph::makeFakeQuantizeTypeRelaxed(
            add,
            element::f32,
            FakeQuantizeOnData{256ul, Shape{}, {0}, {2.55f}, {0}, {2.55f}, precisionBeforeDequantization});

    auto deqStructureBefore = dequantizationBefore;
    deqStructureBefore.multiply.outPrecision = inputPrecision;
    const auto dequantizationBeforeStructure = makeDequantization(FQAfterAdd, deqStructureBefore);
    std::shared_ptr<Node> assign;
    if (opsetVersion == 6) {
        assign = std::make_shared<opset6::Assign>(dequantizationBeforeStructure, variable);
    } else {
        assign = std::make_shared<opset3::Assign>(dequantizationBeforeStructure, "id");
    }
    assign->add_control_dependency(readValue);
    add->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    ngraph::SinkVector sinks{ as_type_ptr<ov::op::Sink>(assign) };
    return std::make_shared<ngraph::Function>(results, sinks, ngraph::ParameterVector{ input }, "AssignAndReadValueFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
