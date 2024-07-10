// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"

#include "typedefs.hpp"


namespace ov {
namespace mlir {

using namespace ::mlir;

Location createLayerLocation(MLIRContext* ctx, const std::string& layerName, const std::string& layerType);

SmallVector<int64_t> importShape(const ov::PartialShape& shape);

Type importPrecision(MLIRContext* ctx, const ov::element::Type& precision);

RankedTensorType importTensor(MLIRContext* ctx,
                                    const ov::PartialShape& shape,
                                    const ov::element::Type& elemType);

Location createLocation(MLIRContext* ctx, NodePtr node);

} // namespace mlir
} // namespace ov