// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../convert_common.hpp"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace ov {
namespace mlir {

template <typename MlirUnaryOp>
struct ConvertUnaryEltwise {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto ov_out_el_ty = node->get_output_element_type(0);
        const auto ov_out_shape = node->get_output_partial_shape(0);
        auto out_ty = importTensor(context.context, ov_out_shape, ov_out_el_ty);
        auto dims = context.get_dynamic_dimension_values(ov_out_shape);
        auto empty = tensor::EmptyOp::create(builder, loc, out_ty, dims);
        return MlirUnaryOp::create(builder, loc, ::mlir::ValueRange{input}, ::mlir::ValueRange{empty});
    }
};

}  // namespace mlir
}  // namespace ov
