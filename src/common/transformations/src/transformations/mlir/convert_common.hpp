// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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

bool elementwise_no_broadcast_predicate_impl(const ov::Output<ov::Node>& output, ov::element::Type type);

template <ov::element::Type_t type>
bool elementwise_no_broadcast_predicate(const ov::Output<ov::Node>& output) {
    return elementwise_no_broadcast_predicate_impl(output, type);
}

// Borrowed it from TPP-MLIR. FIXME: Do we have a better upstreamed alternative?
template <typename T>
mlir::arith::ConstantOp getConstant(OpBuilder &builder, const ov::element::Type& precision, T value) {
    auto unkLoc = builder.getUnknownLoc();
    TypedAttr attr;
    auto type = importPrecision(builder.getContext(), precision);
    if(precision.is_integral()) {
        attr = builder.getIntegerAttr(type, int64_t(value));
    } else if(precision.is_real()) {
        attr = builder.getFloatAttr(type, double(value));
    }
    assert(attr && "Unsupported ConstantOp type");
    return builder.create<arith::ConstantOp>(unkLoc, type, attr);
}

} // namespace mlir
} // namespace ov