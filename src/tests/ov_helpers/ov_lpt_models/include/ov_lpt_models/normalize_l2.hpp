// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>

#include <ngraph/ngraph.hpp>

#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class NormalizeL2Function {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const std::pair<ngraph::PartialShape, ngraph::Shape>& shapes,
        const ngraph::element::Type precisionOnActivation,
        const std::vector<uint64_t>& axes,
        const bool fuseMultiply,
        const bool shift);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& shape,
        const ngraph::op::EpsMode& epsMode,
        const std::vector<size_t>& axes,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& shape,
        const ngraph::op::EpsMode& epsMode,
        const std::vector<size_t>& axes,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
