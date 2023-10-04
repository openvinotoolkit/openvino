// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include <low_precision/layer_transformation.hpp>
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeOnWeightsAndUnsupportedChildFunction {
public:
static std::shared_ptr<ngraph::Function> get(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision,
    const std::shared_ptr<ov::op::v0::Constant> weights,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
