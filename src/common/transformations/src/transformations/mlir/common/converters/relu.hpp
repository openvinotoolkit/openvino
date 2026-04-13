// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/relu.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"

namespace ov {
namespace mlir {

struct ConvertRelu {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = tensor::EmptyOp::create(builder, loc, outType, dynamic_dimensions);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});
        auto relu =
            linalg::MaxOp::create(builder, loc, mlir::ValueRange{input, fill.getResult(0)}, mlir::ValueRange{empty});
        return relu;
    }
};

}  // namespace mlir
}  // namespace ov
