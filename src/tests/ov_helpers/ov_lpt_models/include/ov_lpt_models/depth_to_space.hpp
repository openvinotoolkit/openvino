// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>

#include <ngraph/opsets/opset1.hpp>
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class DepthToSpaceFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const ov::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize);

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize,
        const ov::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize,
        const ov::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
