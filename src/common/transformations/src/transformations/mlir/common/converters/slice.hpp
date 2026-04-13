// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/slice.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"

namespace ov {
namespace mlir {

struct ConvertSlice {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto start = context.getInputs(node)[1];
        const auto stop = context.getInputs(node)[2];
        const auto step = context.getInputs(node)[3];
        const auto axes = context.getInputs(node)[4];

        const auto ov_index_shape = node->get_input_partial_shape(1);
        const auto ov_index_element_type = node->get_input_element_type(1);
        auto dynamic_index_dims = context.get_dynamic_dimension_values(ov_index_shape);

        auto index_type = importTensor(context.context, ov_index_shape, ov_index_element_type);
        auto empty = tensor::EmptyOp::create(builder, loc, index_type, dynamic_index_dims);

        // TODO: this only works for the all-positive numbers case.
        auto sizes = linalg::SubOp::create(builder, loc, mlir::ValueRange{stop, start}, mlir::ValueRange{empty});
        auto slice = tensor::ExtractSliceOp::create(builder, loc, input, mlir::ValueRange{start}, mlir::ValueRange{sizes.getResults()}, mlir::ValueRange{step});
        return slice;
    }
};

}  // namespace mlir
}  // namespace ov
