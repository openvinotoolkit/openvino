// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

#include "convert_common.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "typedefs.hpp"

namespace ov::intel_gpu::mlir {

using ::mlir::MLIRContext;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::Value;
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

    ConversionContext(mlir::MLIRContext* context, mlir::OpBuilder* block_builder, getInputsFn getInputs, getDimValueFn getDimValue);

    [[nodiscard]] mlir::OpBuilder& builder() const {
        return *block_builder;
    }

    [[nodiscard]] SmallVector<Value> get_dynamic_dimension_values(const PartialShape& shape) const;
};

}  // namespace ov::intel_gpu::mlir