// Copyright (C) 2020 Intel Corporation
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

class ConvertMulOrAddWithDequantizationFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type inputPrecision,
        const std::vector<float>& multiplyConst);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::Shape& inputShape,
        const ngraph::element::Type inputPrecision,
        const std::vector<float>& multiplyConst);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
