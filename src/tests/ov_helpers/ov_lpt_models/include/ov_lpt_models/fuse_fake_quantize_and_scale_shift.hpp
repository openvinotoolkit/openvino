// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "common/fake_quantize_on_data.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FuseFakeQuantizeAndScaleShiftFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
