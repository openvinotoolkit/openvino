// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/partial_shape.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class MultiplyWithOneParentFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantize);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
