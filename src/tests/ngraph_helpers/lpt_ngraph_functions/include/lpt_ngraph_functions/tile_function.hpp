// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class TileFunction {
public:
static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization);

static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::PartialShape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize);

static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
