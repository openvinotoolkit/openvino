// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/add.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FuseFakeQuantizeFunction {
public:
    class Branch {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ngraph::element::Type precisionAfterDequantization;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type precisionBeforeAdd,
        const Add& add,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ngraph::element::Type precisionAfterDequantization,
        const ngraph::element::Type precisionFqOnData,
        const FakeQuantizeOnData& fqOnData);

    static std::shared_ptr<ngraph::Function> getReference(
            const ngraph::Shape& inputShape,
            const ngraph::element::Type precisionBeforeAdd,
            const Add& add,
            const ngraph::element::Type precisionBeforeDequantization,
            const DequantizationOperations& dequantization,
            const ngraph::element::Type precisionAfterDequantization,
            const ngraph::element::Type precisionFqOnData,
            const FakeQuantizeOnData& fqOnData);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::Shape& inputShape,
        const std::vector<Branch>& branches,
        const ngraph::element::Type precisionFqOnData,
        const FakeQuantizeOnData& fqOnData);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
