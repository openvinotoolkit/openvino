// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FuseConvertFunction {
public:
    static std::shared_ptr<ov::Model> get(
            const ov::PartialShape& inputShape,
            const ov::element::Type inputPrecision,
            const ov::builder::subgraph::DequantizationOperations& dequantization,
            const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
            const bool constInput);

    static std::shared_ptr<ov::Model> getWithFQ(
            const ov::PartialShape& inputShape,
            const ov::element::Type inputPrecision,
            const ov::builder::subgraph::DequantizationOperations& dequantization,
            const bool constInput);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
