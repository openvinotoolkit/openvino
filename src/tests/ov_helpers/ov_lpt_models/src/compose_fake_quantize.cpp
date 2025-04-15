// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/compose_fake_quantize.hpp"
#include "low_precision/network_helper.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

    std::shared_ptr<ov::Model> ComposeFakeQuantizeFunction::get(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const ov::builder::subgraph::DequantizationOperations& dequantization1,
        const ov::builder::subgraph::DequantizationOperations& dequantization2) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);

        auto fakeQuantize = makeFakeQuantize(input, precision, fqOnData);

        auto results = ov::ResultVector{};
        if (dequantization1.empty() && dequantization2.empty()) {
            results.push_back(std::make_shared<ov::opset1::Result>(fakeQuantize));
        } else {
            if (!dequantization1.empty()) {
                const auto deq = makeDequantization(fakeQuantize, dequantization1);
                results.push_back(std::make_shared<ov::opset1::Result>(deq));
            }
            if (!dequantization2.empty()) {
                const auto deq = makeDequantization(fakeQuantize, dequantization2);
                results.push_back(std::make_shared<ov::opset1::Result>(deq));
            }
        }

        return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ComposeFakeQuantizeFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
