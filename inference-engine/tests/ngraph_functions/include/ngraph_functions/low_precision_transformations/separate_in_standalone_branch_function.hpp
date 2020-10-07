// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
class SeparateInStandaloneBranchFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const size_t numberOfOperations);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type inputPrecision,
        const ngraph::Shape inputShape,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const size_t numberOfOperations,
        const size_t indexOfTargetOperation);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
