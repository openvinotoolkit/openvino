// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "../conversion_context.hpp"

namespace ov {
namespace mlir {

class GatherPattern : public MarkPattern {
public:
    OPENVINO_RTTI("GatherPattern", "0");
    GatherPattern();
};

}  // namespace mlir
}  // namespace ov

