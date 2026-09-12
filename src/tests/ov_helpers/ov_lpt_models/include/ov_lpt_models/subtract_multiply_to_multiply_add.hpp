// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class SubtractMultiplyToMultiplyAddFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::PartialShape& inputShape,
        const ov::element::Type precision,
        const ov::builder::subgraph::FakeQuantizeOnData& fqOnData);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
