// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class ComposeFakeQuantizeFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        const ov::builder::subgraph::DequantizationOperations& dequantization1,
        const ov::builder::subgraph::DequantizationOperations& dequantization2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
