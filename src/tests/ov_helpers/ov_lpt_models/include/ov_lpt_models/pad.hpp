// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class PadFunction {
public:
static std::shared_ptr<ov::Model> get(
    const PartialShape& inputShape,
    const element::Type precisionBeforeDequantization,
    const builder::subgraph::DequantizationOperations& dequantizationBefore,
    const std::vector<int64_t>& padsBegin,
    const std::vector<int64_t>& padsEnd,
    const op::PadMode mode,
    const float padValue,
    const element::Type precisionAfterOperation,
    const builder::subgraph::DequantizationOperations& dequantizationAfter);

static std::shared_ptr<ov::Model> get(
    const PartialShape& inputShape,
    const element::Type inputPrecision,
    const builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<int64_t>& padsBegin,
    const std::vector<int64_t>& padsEnd,
    const op::PadMode mode,
    const float padValue = 0.f);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
