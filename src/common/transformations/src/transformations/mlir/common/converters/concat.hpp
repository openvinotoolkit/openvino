// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/concat.hpp>
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"

namespace ov {
namespace mlir {

struct ConvertConcat {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);

        const auto ov_element_type = node->get_input_element_type(0);
        const auto src_partial_shape = node->get_input_partial_shape(0);
        const auto rank = src_partial_shape.rank().get_length();

        auto concat_node = std::dynamic_pointer_cast<ov::opset1::Concat>(node);
        int64_t axis = concat_node->get_axis();
        if (axis < 0) {
            axis += rank;
        }

        auto concat = tensor::ConcatOp::create(builder, loc, axis, mlir::ValueRange{inputs});
        return concat;
    }
};

}  // namespace mlir
}  // namespace ov

