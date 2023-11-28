// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MatMulWithOptimizedConstantFakeQuantizeFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape1,
        const ov::PartialShape& inputShape2,
        const FakeQuantizeOnData& fqOnData,
        const FakeQuantizeOnData& fqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
