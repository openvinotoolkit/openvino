// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class SubtractFunction {
public:
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
