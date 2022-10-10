// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"



namespace ngraph {
namespace builder {
namespace subgraph {

class SplitFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const element::Type& precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int64_t splitedAxis,
        const size_t numSplits);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const int64_t splitedAxis,
        const size_t numSplit);

    static std::shared_ptr<ngraph::Function> getReference(
        const element::Type& precision,
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type inputPrecision,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const std::vector<ngraph::builder::subgraph::DequantizationOperations>& dequantizationAfter,
        const int64_t splitedAxis,
        const size_t numSplits);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
