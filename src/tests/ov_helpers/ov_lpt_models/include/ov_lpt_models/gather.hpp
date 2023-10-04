// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class GatherFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::PartialShape& inputShape,
        const std::vector<size_t>& gatherIndicesShape,
        const std::vector<int>& gatherIndicesValues,
        const std::vector<int>& axis,
        const int64_t batch_dims,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization,
        const int opset_version);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::PartialShape& inputShape,
        const std::vector<size_t>& gatherIndicesShape,
        const std::vector<int>& gatherIndicesValues,
        const std::vector<int>& axis,
        const int64_t batch_dims,
        const ngraph::element::Type precisionBeforeFq,
        const FakeQuantizeOnData& fqOnData,
        const int opset_version);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::PartialShape& inputShape,
        const std::vector<size_t>& gatherIndicesShape,
        const std::vector<int>& gatherIndicesValues,
        const std::vector<int>& axis,
        const int64_t batch_dims,
        const ngraph::element::Type precisionBeforeDequantization,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const int opset_version);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
