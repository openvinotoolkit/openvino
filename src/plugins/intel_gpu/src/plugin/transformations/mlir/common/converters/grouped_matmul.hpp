// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../convert_common.hpp"
#include "compressed/weights_decompression.hpp"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "openvino/op/grouped_matmul.hpp"
#include "ov_ops/grouped_matmul_compressed.hpp"

namespace ov::intel_gpu::mlir {

// GroupedMatMul-17 -> linalgx.grouped_matmul. Only the 2Dx3D (MoE) case is supported:
//   A [total, K] x B [E, N, K], offsets [E] -> out [total, N]
// The 3Dx3D case is rejected by the pattern and left to the cldnn path.
//
// The internal GroupedMatMulCompressed (2Dx3D form) is supported too:
//   A [total, K] x B [E, N, K] (u4/i4/u8/i8), offsets [E], scale [E, N, G], (zp [E, N, G])
// The weights are dequantized by an elementwise linalg.generic which feeds linalgx.grouped_matmul.
// Group-wise scale / zp are requested in the [E, G, N] physical order.
struct ConvertGroupedMatMul {
    Operation* operator()(ConversionContext& context, const NodePtr& node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        auto* ctx = context.context;

        const auto inputs = context.getInputs(node);
        const bool compressed = ov::is_type<ov::op::internal::GroupedMatMulCompressed>(node);
        if (compressed) {
            OPENVINO_ASSERT(inputs.size() == 4 || inputs.size() == 5, "GroupedMatMulCompressed: the 2Dx3D case expects 4 or 5 inputs, got ", inputs.size());
        } else {
            OPENVINO_ASSERT(inputs.size() == 3, "GroupedMatMul: the 2Dx3D case expects 3 inputs, got ", inputs.size());
        }

        const Value lhs = inputs[0];      // [total, K]
        Value weights = inputs[1];        // [E, N, K]
        const Value offsets = inputs[2];  // [E]

        const auto ov_out_type = node->get_output_element_type(0);
        const auto& out_shape = node->get_output_partial_shape(0);
        const auto out_type = importTensor(ctx, out_shape, ov_out_type);

        if (compressed) {
            const bool has_zp = inputs.size() > 4;
            const decompression::CompressedWeights cw{/*weights=*/weights,
                                                      /*weights_signed=*/node->get_input_element_type(1).is_signed(),
                                                      /*scale=*/inputs[3],
                                                      /*zero_point=*/has_zp ? inputs[4] : Value{},
                                                      /*zp_signed=*/has_zp && node->get_input_element_type(4).is_signed()};
            for (size_t i = 3; i < inputs.size(); ++i) {
                if (auto order = decompression::grouped_param_order(decompression::static_shape(inputs[i]))) {
                    context.requestInputOrder(node->input(i), std::move(*order));
                }
            }
            weights = decompression::decompress_weights(context, loc, cw, ov_out_type);
        }

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
