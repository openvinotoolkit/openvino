// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FoldFakeQuantizeFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::Shape& constShape,
        const std::vector<float>& constValues,
        const FakeQuantizeOnData& fakeQuantizeOnData);

    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::Shape& constShape,
        const std::vector<float>& constValues);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
