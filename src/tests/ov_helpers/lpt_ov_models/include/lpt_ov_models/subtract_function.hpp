// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>


#include "lpt_ov_models/common/dequantization_operations.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "lpt_ov_models/common/builders.hpp"

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
