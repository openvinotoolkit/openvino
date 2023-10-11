// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/ngraph.hpp>

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
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeAdd,
        const Add& add,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ngraph::element::Type precisionAfterDequantization,
        const ngraph::element::Type precisionFqOnData,
        const FakeQuantizeOnDataWithConstant& fqOnData);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBeforeAdd,
        const Add& add,
        const ngraph::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ngraph::element::Type precisionAfterDequantization,
        const ngraph::element::Type precisionFqOnData,
        const FakeQuantizeOnDataWithConstant& fqOnData);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::PartialShape& inputShape,
        const ngraph::element::Type precisionBefore,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::PartialShape& inputShape,
        const std::vector<Branch>& branches,
        const ngraph::element::Type precisionFqOnData,
        const FakeQuantizeOnData& fqOnData);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
