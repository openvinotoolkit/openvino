// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulWithOptimizedConstantFakeQuantizeFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape1,
        const ngraph::PartialShape& inputShape2,
        const FakeQuantizeOnData& fqOnData,
        const FakeQuantizeOnData& fqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
