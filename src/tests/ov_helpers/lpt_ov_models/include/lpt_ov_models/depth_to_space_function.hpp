// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>


#include <openvino/opsets/opset1.hpp>
#include "lpt_ov_models/common/dequantization_operations.hpp"

namespace ov {
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
        const ov::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize,
        const ov::element::Type precisionBeforeDequantization,
        const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
