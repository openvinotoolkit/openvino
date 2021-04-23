// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>

#include "low_precision/network_helper.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

#include "lpt_ngraph_functions/shuffle_channels_function.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<Function> ShuffleChannelsFunction::getOriginal(
    const element::Type inputPrecision,
    const Shape& inputShape,
    const builder::subgraph::DequantizationOperations& deqBefore,
    const std::int64_t axis,
    const std::int64_t group) {
    const auto input = std::make_shared<opset1::Parameter>(inputPrecision, inputShape);
    const auto dequantization = makeDequantization(input, deqBefore);

    const auto shuffleChannels = std::make_shared<opset1::ShuffleChannels>(dequantization, axis, group);
    shuffleChannels->set_friendly_name("output");

    const auto function = std::make_shared<ngraph::Function>(
        ResultVector{ std::make_shared<opset1::Result>(shuffleChannels) },
        ParameterVector{ input },
        "ShuffleChannelsFunction");

    return function;
}

std::shared_ptr<ngraph::Function> ShuffleChannelsFunction::getOriginal(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
    const std::int64_t axis,
    const std::int64_t group) {
    const auto input = std::make_shared<opset1::Parameter>(inputPrecision, inputShape);
    const auto fakeQuantize = makeFakeQuantize(input, inputPrecision, fqOnData);

    const auto shuffleChannels = std::make_shared<opset1::ShuffleChannels>(fakeQuantize, axis, group);
    shuffleChannels->set_friendly_name("output");

    const auto function = std::make_shared<ngraph::Function>(
        ResultVector{ std::make_shared<opset1::Result>(shuffleChannels) },
        ParameterVector{ input },
        "ShuffleChannelsFunction");

    return function;
}

std::shared_ptr<ngraph::Function> ShuffleChannelsFunction::getReference(
    const ngraph::element::Type inputPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::DequantizationOperations& deqBefore,
    const std::int64_t axis,
    const std::int64_t group,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& deqAfter) {
    const auto input = std::make_shared<opset1::Parameter>(inputPrecision, inputShape);
    const auto dequantizationBefore = makeDequantization(input, deqBefore);

    const auto shuffleChannels = std::make_shared<opset1::ShuffleChannels>(dequantizationBefore, axis, group);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(shuffleChannels, precisionAfterOperation);

    const auto dequantizationAfter = makeDequantization(shuffleChannels, deqAfter);
    dequantizationAfter->set_friendly_name("output");

    const auto function = std::make_shared<ngraph::Function>(
        ResultVector{ std::make_shared<opset1::Result>(dequantizationAfter) },
        ParameterVector{ input },
        "ShuffleChannelsFunction");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
