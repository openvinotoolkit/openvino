// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/compose_fake_quantize_function.hpp"
#include "low_precision/network_helper.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

    std::shared_ptr<ngraph::Function> ComposeFakeQuantizeFunction::get(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization2) {
        const auto input = std::make_shared<ngraph::op::v0::Parameter>(precision, inputShape);

        auto fakeQuantize = makeFakeQuantize(input, precision, fqOnData);

        auto results = ngraph::ResultVector{};
        if (dequantization1.empty() && dequantization2.empty()) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(fakeQuantize));
        } else {
            if (!dequantization1.empty()) {
                const auto deq = makeDequantization(fakeQuantize, dequantization1);
                results.push_back(std::make_shared<ngraph::opset1::Result>(deq));
            }
            if (!dequantization2.empty()) {
                const auto deq = makeDequantization(fakeQuantize, dequantization2);
                results.push_back(std::make_shared<ngraph::opset1::Result>(deq));
            }
        }

        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ComposeFakeQuantizeFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
