// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>


#include "lpt_ov_models/common/dequantization_operations.hpp"
#include "ov_models/subgraph_builders.hpp"

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
