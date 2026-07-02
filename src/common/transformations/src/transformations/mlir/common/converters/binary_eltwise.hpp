// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/relu.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace mlir {

using namespace ov;
using namespace ov::mlir;
using ::mlir::ValueRange;

template<typename MlirBinOpBuilder>
struct ConvertBinaryEltwise {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        const int output_rank = ov_output_shape.rank().get_length();

        SmallVector<Value> dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);

        SmallVector<Value> broadcasted_inputs;
        for(size_t i = 0; i < inputs.size(); ++i) {
            auto [collapse_groups, dimensions] = broadcast_dimensions(node->get_input_partial_shape(i), ov_output_shape);
            if(!dimensions.empty()) {
                // FIXME: Find a way to avoid dimension squeezing before applying linalg.broadcast
                // Step 1: Squeeze input shape to eliminate broadcasted dimensions
                auto squeezed = tensor::CollapseShapeOp::create(builder, loc, inputs[i], collapse_groups);
                // Step 2: Broadcast squeezed shape to the target shape
                auto empty = tensor::EmptyOp::create(builder, loc, outType, dynamic_dimensions);
                auto op = linalg::BroadcastOp::create(builder, loc, squeezed, empty, dimensions);
                broadcasted_inputs.push_back(op.getResult()[0]);
            } else {
                broadcasted_inputs.push_back(inputs[i]);
            }
        }

        auto empty = tensor::EmptyOp::create(builder, loc, outType, dynamic_dimensions);
        auto op = MlirBinOpBuilder::create(builder, loc, ValueRange(broadcasted_inputs), ValueRange{empty});
        return op;
    }
};

}  // namespace mlir
}  // namespace ov
