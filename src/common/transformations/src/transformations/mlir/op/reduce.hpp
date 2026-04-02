// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../conversion_context.hpp"

namespace ov {
namespace mlir {

template <typename OVOp>
class ReducePattern : public MarkPattern {
public:
    OPENVINO_RTTI("ReducePattern", "0");
    ReducePattern();
};

}  // namespace mlir
}  // namespace ov
