// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class PadFunction {
public:
static std::shared_ptr<ngraph::Function> get(
    const PartialShape& inputShape,
    const element::Type precisionBeforeDequantization,
    const builder::subgraph::DequantizationOperations& dequantizationBefore,
    const std::vector<uint64_t>& padsBegin,
    const std::vector<uint64_t>& padsEnd,
    const op::PadMode mode,
    const float padValue,
    const element::Type precisionAfterOperation,
    const builder::subgraph::DequantizationOperations& dequantizationAfter);

static std::shared_ptr<Function> get(
    const PartialShape& inputShape,
    const element::Type inputPrecision,
    const builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<uint64_t>& padsBegin,
    const std::vector<uint64_t>& padsEnd,
    const op::PadMode mode,
    const float padValue = 0.f);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
