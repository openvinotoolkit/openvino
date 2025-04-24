// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <low_precision/layer_transformation.hpp>
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FakeQuantizeOnWeightsAndUnsupportedChildFunction {
public:
static std::shared_ptr<ov::Model> get(
    const ov::Shape& inputShape,
    const ov::element::Type inputPrecision,
    const std::shared_ptr<ov::op::v0::Constant> weights,
    const ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
