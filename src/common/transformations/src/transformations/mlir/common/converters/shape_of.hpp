// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <openvino/op/shape_of.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"


namespace ov {
namespace mlir {

struct ConvertShapeOf {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        const auto input = context.getInputs(node)[0];
        auto shapeOf = shape::ShapeOfOp::create(builder, loc, mlir::ValueRange{input});
        auto casted_type = RankedTensorType::get(ArrayRef(importShape(ov_output_shape)), importPrecision(context.context, ov_output_element_type));
        auto cast = arith::IndexCastOp::create(builder, loc, casted_type, mlir::ValueRange{shapeOf});
        return cast;
    }
};

}  // namespace mlir
}  // namespace ov
