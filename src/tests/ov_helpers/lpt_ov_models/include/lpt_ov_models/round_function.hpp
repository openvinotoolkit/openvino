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

class RoundWithToleranceFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantization);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantization);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
