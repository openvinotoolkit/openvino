// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>


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
        ov::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
        ov::element::Type precisionAfterDequantization;
    };

    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeAdd,
        const Add& add,
        const ov::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ov::element::Type precisionAfterDequantization,
        const ov::element::Type precisionFqOnData,
        const FakeQuantizeOnDataWithConstant& fqOnData);

    static std::shared_ptr<ov::Model> getReference(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeAdd,
        const Add& add,
        const ov::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ov::element::Type precisionAfterDequantization,
        const ov::element::Type precisionFqOnData,
        const FakeQuantizeOnDataWithConstant& fqOnData);

    static std::shared_ptr<ov::Model> get(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBefore,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ov::Model> get(
        const ov::PartialShape& inputShape,
        const std::vector<Branch>& branches,
        const ov::element::Type precisionFqOnData,
        const FakeQuantizeOnData& fqOnData);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
