// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinTypes.h"

#include "conversion_context.hpp"


namespace ov {
namespace mlir {

using namespace ::mlir;

ConversionContext::ConversionContext(
    mlir::MLIRContext* context, mlir::OpBuilder* block_builder,
    getInputsFn getInputs, getDimValueFn getDimValue
)
    : context(context),
        block_builder(block_builder),
        getInputs(getInputs),
        getDimValue(getDimValue) {}


SmallVector<Value> ConversionContext::get_dynamic_dimension_values (const PartialShape& shape) {
    SmallVector<Value> dims;
    for (const auto& dim: shape) {
        if (dim.is_dynamic()) {
            dims.push_back(getDimValue(dim));
        }
    }
    return dims;
}


} // namespace mlir
} // namespace ov