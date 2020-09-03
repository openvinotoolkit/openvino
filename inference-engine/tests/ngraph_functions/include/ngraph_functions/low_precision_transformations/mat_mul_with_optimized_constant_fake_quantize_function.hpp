// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulWithOptimizedConstantFakeQuantizeFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData,
        const FakeQuantizeOnData& fqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
