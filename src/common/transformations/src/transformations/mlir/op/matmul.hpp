// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/Value.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

#include "../conversion_context.hpp"

namespace ov {
namespace mlir {

class MatMulPattern : public MarkPattern {
public:
    OPENVINO_RTTI("MatMulPattern", "0");
    MatMulPattern();
};


} // namespace mlir
} // namespace ov