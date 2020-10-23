// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ReshapeFullyConnectedFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type inputPrecision1,
        const ngraph::element::Type inputPrecision2,
        const ngraph::element::Type inputPrecision3,
        const ngraph::Shape& outputShape,
        const ngraph::element::Type outputPrecision);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type inputPrecision1,
        const ngraph::element::Type inputPrecision2,
        const ngraph::element::Type inputPrecision3,
        const ngraph::Shape& outputShape,
        const ngraph::element::Type outputPrecision);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
