// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <low_precision/layer_transformation.hpp>
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class SliceFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type inputPrecision,
        const ov::PartialShape& inputShape,
        const ov::builder::subgraph::FakeQuantizeOnData& fakeQuantize,
        const std::vector<int64_t>& start,
        const std::vector<int64_t>& stop,
        const std::vector<int64_t>& step,
        const std::vector<int64_t>& axes);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
