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

        // TODO: move the unit-dimension-folding logic to the graph-compiler
        bool batch = true;
        auto canCollapse = [](mlir::Value tensor) { // rank <= 2 or all leading dimensions are 1
            auto shape = mlir::cast<RankedTensorType>(tensor.getType()).getShape().drop_back(2);
            return std::all_of(shape.begin(), shape.end(), [](int64_t d) {
                return d == 1;
            });
        };
        if (canCollapse(ins[0]) && canCollapse(ins[1]) && canCollapse(outs[0])) {
            auto collapse = [&](Value tensor) -> Value { // Has no-op if rank <= 2
                auto shape = mlir::cast<RankedTensorType>(tensor.getType()).getShape();
                int64_t rank = shape.size();
                if (rank <= 2)
                    return tensor;
                SmallVector<ReassociationIndices> reassoc;
                ReassociationIndices leading;
                for (int64_t i = 0; i < rank - 1; ++i)
                    leading.push_back(i);
                reassoc.push_back(leading);
                reassoc.push_back(ReassociationIndices{rank - 1});
                return tensor::CollapseShapeOp::create(builder, loc, tensor, reassoc).getResult();
            };
            ins[0] = collapse(ins[0]);
            ins[1] = collapse(ins[1]);
            outs[0] = collapse(outs[0]);
            batch = false;
        }

        auto expand = [&](Value tensor) -> Value {
            int64_t rank = ov_output_shape.size();
            auto type = mlir::cast<RankedTensorType>(tensor.getType());
            auto shape = type.getShape();
            if (shape.size() == rank)
                return tensor;
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices leading;
            for (int64_t i = 0; i < rank - 1; ++i)
                leading.push_back(i);
            reassoc.push_back(leading);
            reassoc.push_back(ReassociationIndices{rank - 1});
            SmallVector<int64_t> new_shape(rank, 1);
            std::copy(shape.begin(), shape.end(), new_shape.end() - shape.size());
            auto new_type = mlir::RankedTensorType::get(new_shape, type.getElementType());
            return tensor::ExpandShapeOp::create(builder, loc, new_type, tensor, reassoc).getResult();
        };

        Operation* matmul;
        if (batch) {
            // Expand if required, to have all inputs of the same rank.
            ins[0] = expand(ins[0]);
            ins[1] = expand(ins[1]);
            if (isTransposedA) {
                matmul = linalg::BatchMatmulTransposeAOp::create(builder, loc, ins, outs);
            } else if (isTransposedB) {
                matmul = linalg::BatchMatmulTransposeBOp::create(builder, loc, ins, outs);
            } else {
                matmul = linalg::BatchMatmulOp::create(builder, loc, ins, outs);
            }
        } else {
            if (isTransposedA) {
                matmul = linalg::MatmulTransposeAOp::create(builder, loc, ins, outs);
            } else if (isTransposedB) {
                matmul = linalg::MatmulTransposeBOp::create(builder, loc, ins, outs);
            } else {
                matmul = linalg::MatmulOp::create(builder, loc, ins, outs);
            }
            matmul = expand(matmul->getResult(0)).getDefiningOp();
        }
        return matmul;
    }
};

} // namespace mlir
} // namespace ov