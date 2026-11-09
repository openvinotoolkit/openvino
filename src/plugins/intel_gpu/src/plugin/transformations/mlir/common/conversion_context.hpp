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
    // Physical dimension order of a tensor, e.g. {0, 2, 1} for a logical [E, N, G] tensor stored as [E, G, N].
    using DimOrder = SmallVector<int64_t>;
    using requestInputOrderFn = std::function<void(const ov::Input<ov::Node>&, DimOrder)>;
    using NodeOutputMap = std::map<ov::Output<ov::Node>, mlir::Value>;

    mlir::MLIRContext* context;
    mlir::OpBuilder* block_builder;
    getInputsFn getInputs;
    getDimValueFn getDimValue;
    // Asks for the source of `input` to be passed in the given physical order. This is only a hint that
    // the driver may ignore, so the IR built by a converter must stay valid for any order.
    requestInputOrderFn requestInputOrder;

    ConversionContext(mlir::MLIRContext* context,
                      mlir::OpBuilder* block_builder,
                      getInputsFn getInputs,
                      getDimValueFn getDimValue,
                      requestInputOrderFn requestInputOrder);

    [[nodiscard]] mlir::OpBuilder& builder() const {
        return *block_builder;
    }

    [[nodiscard]] SmallVector<Value> get_dynamic_dimension_values(const PartialShape& shape) const;
};

}  // namespace ov::intel_gpu::mlir