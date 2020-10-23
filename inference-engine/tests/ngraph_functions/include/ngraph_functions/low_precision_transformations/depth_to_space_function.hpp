// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class DepthToSpaceFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::Shape& inputShape,
        const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
