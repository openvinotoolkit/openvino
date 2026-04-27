// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/transpose.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"

namespace ov {
namespace mlir {

struct ConvertTranspose {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        // TODO: support dynamic inputs
        // const auto order = context.getInputs(node)[1];

        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto out_type = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);

        auto const_order = dynamic_cast<ov::op::v0::Constant*>(node->get_input_node_ptr(1));
        assert(const_order && "non-const order not supported");
        ov::Coordinate coords = const_order->get_coordinate_val();
        SmallVector<int64_t> order(coords.begin(), coords.end());

        auto empty = tensor::EmptyOp::create(builder, loc, out_type, dynamic_dimensions);
        auto transpose = linalg::TransposeOp::create(builder, loc, input, empty, order);
        return transpose;
    }
};

}  // namespace mlir
}  // namespace ov

