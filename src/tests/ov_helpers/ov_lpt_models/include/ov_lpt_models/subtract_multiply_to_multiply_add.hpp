// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "ov_lpt_models/common/add.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/multiply.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class SubtractMultiplyToMultiplyAddFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const ngraph::element::Type precisionAfterDequantization);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precision,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const ngraph::element::Type precisionAfterDequantization,
        const ngraph::builder::subgraph::Multiply& multiply,
        const ngraph::builder::subgraph::Add& add);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
