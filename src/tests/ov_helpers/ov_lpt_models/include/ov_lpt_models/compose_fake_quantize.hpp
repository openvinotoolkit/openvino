// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_models/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ComposeFakeQuantizeFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
