// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/grouped_matmul.hpp"

#include "../convert_common.hpp"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"

namespace ov::intel_gpu::mlir {

// GroupedMatMul-17 -> linalgx.grouped_matmul. Only the 2Dx3D (MoE) case is supported:
//   A [total, K] x B [E, N, K], offsets [E] -> out [total, N]
// The 3Dx3D case is rejected by the pattern and left to the cldnn path.
struct ConvertGroupedMatMul {
    Operation* operator()(ConversionContext& context, const NodePtr& node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        auto* ctx = context.context;

        const auto inputs = context.getInputs(node);
        OPENVINO_ASSERT(inputs.size() == 3, "GroupedMatMul: the 2Dx3D case expects 3 inputs, got ", inputs.size());

        const Value lhs = inputs[0];      // [total, K]
        const Value weights = inputs[1];  // [E, N, K]
        const Value offsets = inputs[2];  // [E]

        const auto ov_out_type = node->get_output_element_type(0);
        const auto& out_shape = node->get_output_partial_shape(0);
        const auto out_type = importTensor(ctx, out_shape, ov_out_type);

        // The op accumulates into its init, so zero it.
        Value empty = tensor::EmptyOp::create(builder, loc, out_type, context.get_dynamic_dimension_values(out_shape)).getResult();
        Value zero = getConstant(builder, ov_out_type, 0).getResult();
        Value init = linalg::FillOp::create(builder, loc, ValueRange{zero}, ValueRange{empty}).getResult(0);

        // Domain: (expert, m, n, k).
        AffineExpr e_dim, m_dim, n_dim, k_dim;
        bindDims(ctx, e_dim, m_dim, n_dim, k_dim);
        const SmallVector<AffineMap> maps = {
            AffineMap::get(4, 0, {m_dim, k_dim}, ctx),         // lhs     [total, K]
            AffineMap::get(4, 0, {e_dim, n_dim, k_dim}, ctx),  // weights [E, N, K]
            AffineMap::get(4, 0, {e_dim}, ctx),                // offsets [E]
            AffineMap::get(4, 0, {m_dim, n_dim}, ctx),         // output  [total, N]
        };

        return linalgx::GroupedMatmulOp::create(builder,
                                                loc,
                                                TypeRange{out_type},
                                                lhs,
                                                weights,
                                                offsets,
                                                // Optional perf hint (rows-per-group bound, as in oneDNN); not available from ov::opset yet.
                                                /*max_group_size=*/Value{},
                                                init,
                                                builder.getAffineMapArrayAttr(maps));
    }
};

}  // namespace ov::intel_gpu::mlir
