// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/matmul.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"


namespace ov {
namespace mlir {

struct ConvertMatMul {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        // TODO: Support broadcasts
        const auto inputs = context.getInputs(node);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = tensor::EmptyOp::create(builder, loc, outType, dynamic_dimensions);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});

        mlir::SmallVector<Value, 2> ins{inputs[0], inputs[1]};
        mlir::SmallVector<Value, 1> outs{fill.getResult(0)};

        auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        assert(matmul_node);
        bool isTransposedA = matmul_node->get_transpose_a();
        bool isTransposedB = matmul_node->get_transpose_b();
        assert(!(isTransposedA && isTransposedB));

        Operation* matmul;
        if (isTransposedA) {
            matmul = linalg::MatmulTransposeAOp::create(builder, loc, ins, outs);
        } else if (isTransposedB) {
            matmul = linalg::MatmulTransposeBOp::create(builder, loc, ins, outs);
        } else {
            matmul = linalg::MatmulOp::create(builder, loc, ins, outs);
        }

        return matmul;
    }
};

} // namespace mlir
} // namespace ov