// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

#include "mlir/IR/Value.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

#include "typedefs.hpp"
#include "convert_common.hpp"

namespace ov {
namespace mlir {

using ::mlir::Value;
using ::mlir::MLIRContext;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::ValueRange;

class ConversionContext {
public:
    using getInputsFn = std::function<SmallVector<mlir::Value>(NodePtr)>;
    using getDimValueFn = std::function<Value(const Dimension&)>;
    using NodeOutputMap = std::map<ov::Output<ov::Node>, mlir::Value>;

    mlir::MLIRContext* context;
    mlir::OpBuilder* block_builder;
    getInputsFn getInputs;
    getDimValueFn getDimValue;

    ConversionContext(
        mlir::MLIRContext* context, mlir::OpBuilder* block_builder,
        getInputsFn getInputs, getDimValueFn getDimValue
    );

    mlir::OpBuilder& builder() {
        return *block_builder;
    }

    SmallVector<Value> get_dynamic_dimension_values(const PartialShape& shape);
};

} // namespace mlir
} // namespace ov