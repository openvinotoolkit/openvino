// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulWithOptimizedConstantFakeQuantizeFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape1,
        const ngraph::Shape& inputShape2,
        const FakeQuantizeOnData& fqOnData,
        const FakeQuantizeOnData& fqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
