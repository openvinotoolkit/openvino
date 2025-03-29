// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

#include "ov_lpt_models/shuffle_channels.hpp"

namespace ov {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> ShuffleChannelsFunction::getOriginal(
    const ov::element::Type inputPrecision,
    const PartialShape& inputShape,
    const builder::subgraph::DequantizationOperations& deqBefore,
    const std::int64_t axis,
    const std::int64_t group) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    const auto dequantization = makeDequantization(input, deqBefore);

    const auto shuffleChannels = std::make_shared<ov::opset1::ShuffleChannels>(dequantization, axis, group);
    shuffleChannels->set_friendly_name("output");

    const auto function =
        std::make_shared<ov::Model>(ResultVector{std::make_shared<ov::opset1::Result>(shuffleChannels)},
                                    ov::ParameterVector{input},
                                    "ShuffleChannelsFunction");

    return function;
}

std::shared_ptr<ov::Model> ShuffleChannelsFunction::getOriginal(
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::FakeQuantizeOnData& fqOnData,
    const std::int64_t axis,
    const std::int64_t group) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    const auto fakeQuantize = makeFakeQuantize(input, inputPrecision, fqOnData);

    const auto shuffleChannels = std::make_shared<ov::opset1::ShuffleChannels>(fakeQuantize, axis, group);
    shuffleChannels->set_friendly_name("output");

    const auto function =
        std::make_shared<ov::Model>(ResultVector{std::make_shared<ov::opset1::Result>(shuffleChannels)},
                                    ov::ParameterVector{input},
                                    "ShuffleChannelsFunction");

    return function;
}

std::shared_ptr<ov::Model> ShuffleChannelsFunction::getReference(
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const ov::builder::subgraph::DequantizationOperations& deqBefore,
    const std::int64_t axis,
    const std::int64_t group,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& deqAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    const auto dequantizationBefore = makeDequantization(input, deqBefore);

    const auto shuffleChannels = std::make_shared<ov::opset1::ShuffleChannels>(dequantizationBefore, axis, group);
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(shuffleChannels, precisionAfterOperation);

    const auto dequantizationAfter = makeDequantization(shuffleChannels, deqAfter);
    dequantizationAfter->set_friendly_name("output");

    const auto function =
        std::make_shared<ov::Model>(ResultVector{std::make_shared<ov::opset1::Result>(dequantizationAfter)},
                                    ov::ParameterVector{input},
                                    "ShuffleChannelsFunction");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
