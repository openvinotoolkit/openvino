// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FuseConvertFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
            const ngraph::Shape& inputShape,
            const ngraph::element::Type inputPrecision,
            const ngraph::builder::subgraph::DequantizationOperations& dequantization,
            const bool constInput);

    static std::shared_ptr<ngraph::Function> getWithFQ(
            const ngraph::Shape& inputShape,
            const ngraph::element::Type inputPrecision,
            const ngraph::builder::subgraph::DequantizationOperations& dequantization,
            const bool constInput);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
