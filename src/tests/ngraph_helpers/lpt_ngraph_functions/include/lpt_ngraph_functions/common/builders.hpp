// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/ops.hpp>
#include <ngraph/op/constant.hpp>
#include "ngraph_ops/type_relaxed.hpp"

#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/add.hpp"
#include "lpt_ngraph_functions/common/convolution.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/reshape.hpp"
#include "lpt_ngraph_functions/common/transpose.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

template <typename Operation, typename OperationDesc>
std::shared_ptr<Node> makeElementwise(const std::shared_ptr<ngraph::Node> data, const OperationDesc& description) {
    std::vector<size_t> shape;
    if (description.constantShapeIsDefined) {
        shape = description.constantShape;
    } else {
        if (description.values.size() == 1ul) {
            shape = std::vector<size_t>({});
        } else {
            shape = std::vector<size_t>(data->get_output_partial_shape(0).rank().get_length(), 1ul);
            shape[shape.size() >= 2 ? 1ul : 0] = description.values.size();
        }
    }

    const auto operationConst = std::make_shared<ngraph::opset1::Constant>(
        description.outPrecision,
        shape,
        description.values);

    std::shared_ptr<Operation> operation;
    if ((description.outPrecision == element::undefined) || (description.outPrecision == data->get_output_element_type(0))) {
        operation = std::make_shared<Operation>(data, operationConst);
    } else {
        operation = std::make_shared<op::TypeRelaxed<Operation>>(
            std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(data, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(operationConst, element::f32).get());
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(operation, description.outPrecision);
    }

    if (ov::is_type<ngraph::opset1::Subtract>(operation) || ov::is_type<ngraph::opset1::Add>(operation)) {
        replace_node(
            operationConst,
            ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(operationConst, data->get_output_element_type(0)));
    }

    return operation;
}

std::shared_ptr<Node> makeDequantization(
    const Output<Node>& data,
    const DequantizationOperations& dequantizationOperations);

std::shared_ptr<Node> makeMultiply(const Output<Node>& data, const DequantizationOperations::Multiply& multiply);

std::shared_ptr<Node> makeReshape(const Output<Node>& data, const Reshape& reshape);

std::shared_ptr<Node> makeTranspose(const Output<Node>& data, const Transpose& reshape);

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(
    const Output<Node>& output,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData);

std::shared_ptr<ngraph::opset1::Convolution> makeConvolution(const Output<Node>& output, const Convolution& convolution);

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantizeTypeRelaxed(
    const Output<ngraph::Node>& output,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData);

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(
    const Output<Node>& input,
    const ngraph::element::Type constantPrecision,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const bool subgraphOnConstantPath = false);

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type constantPrecision,
    const FakeQuantizeOnDataWithConstant& fqOnData);

void addAttributes(std::vector<std::shared_ptr<ngraph::Node>> nodes, std::vector<ov::Any> attributes);

std::shared_ptr<Node> makeConvolution(
    const std::shared_ptr<Node>& parent,
    const element::Type precision,
    const bool weightsWithoutFQ,
    const element::Type weightsprecision = element::i8);

} // namespace subgraph
} // namespace builder
} // namespace ngraph
