// Copyright (C) 2018-2023 Intel Corporation
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

class FuseConvertFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
            const ngraph::PartialShape& inputShape,
            const ngraph::element::Type inputPrecision,
            const ngraph::builder::subgraph::DequantizationOperations& dequantization,
            const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
            const bool constInput);

    static std::shared_ptr<ngraph::Function> getWithFQ(
            const ngraph::PartialShape& inputShape,
            const ngraph::element::Type inputPrecision,
            const ngraph::builder::subgraph::DequantizationOperations& dequantization,
            const bool constInput);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
