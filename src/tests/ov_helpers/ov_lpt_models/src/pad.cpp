// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/opsets/opset12.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/pad.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> PadFunction::get(const PartialShape& inputShape,
                                            const element::Type precisionBeforeDequantization,
                                            const builder::subgraph::DequantizationOperations& dequantizationBefore,
                                            const std::vector<int64_t>& padsBegin,
                                            const std::vector<int64_t>& padsEnd,
                                            const op::PadMode mode,
                                            const float padValue,
                                            const element::Type precisionAfterOperation,
                                            const builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto padsBeginConst = ov::opset1::Constant::create(ov::element::i64, Shape{padsBegin.size()}, padsBegin);
    const auto padsEndConst = ov::opset1::Constant::create(ov::element::i64, Shape{padsEnd.size()}, padsEnd);
    const auto padsValueConst = ov::opset1::Constant::create(deqBefore->get_output_element_type(0), Shape{}, { padValue });

    const auto pad = std::make_shared<ov::op::TypeRelaxed<ov::op::v12::Pad>>(
        std::vector<ov::element::Type>{precisionAfterOperation,
                                       ov::element::u64,
                                       ov::element::u64,
                                       precisionAfterOperation},
        std::vector<ov::element::Type>{precisionAfterOperation},
        ov::op::TemporaryReplaceOutputType(deqBefore, precisionAfterOperation).get(),
        ov::op::TemporaryReplaceOutputType(padsBeginConst, ov::element::i64).get(),
        ov::op::TemporaryReplaceOutputType(padsEndConst, ov::element::i64).get(),
        ov::op::TemporaryReplaceOutputType(padsValueConst, precisionAfterOperation).get(),
        mode);

    const auto deqAfter = makeDequantization(pad, dequantizationAfter);
    deqAfter->set_friendly_name("Pad");

    const auto function = std::make_shared<ov::Model>(
        ResultVector{ std::make_shared<ov::opset1::Result>(deqAfter) },
        ov::ParameterVector{ input }, "PadTransformation");

    return function;
}

std::shared_ptr<ov::Model> PadFunction::get(const PartialShape& inputShape,
                                            const element::Type inputPrecision,
                                            const builder::subgraph::FakeQuantizeOnData& fakeQuantize,
                                            const std::vector<int64_t>& padsBegin,
                                            const std::vector<int64_t>& padsEnd,
                                            const op::PadMode mode,
                                            const float padValue) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    const auto fqOnData = makeFakeQuantize(input, inputPrecision, fakeQuantize);

    const auto padsBeginConst = ov::opset1::Constant::create(ov::element::i64, Shape{padsBegin.size()}, padsBegin);
    const auto padsEndConst = ov::opset1::Constant::create(ov::element::i64, Shape{padsEnd.size()}, padsEnd);
    const auto padsValueConst = ov::opset1::Constant::create(inputPrecision, Shape{}, { padValue });
    const auto pad = std::make_shared<ov::op::v12::Pad>(fqOnData, padsBeginConst, padsEndConst, padsValueConst, mode);
    pad->set_friendly_name("Pad");

    const auto function = std::make_shared<ov::Model>(ResultVector{std::make_shared<ov::opset1::Result>(pad)},
                                                      ov::ParameterVector{input},
                                                      "PadTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
