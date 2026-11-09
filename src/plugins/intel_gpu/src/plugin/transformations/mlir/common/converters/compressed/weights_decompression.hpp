// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <numeric>
#include <optional>

#include "../../conversion_context.hpp"
#include "../../convert_common.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/TypeUtilities.h"

// Helpers to dequantize compressed weights of the matmul-like ops (GroupedMatMulCompressed, ...).
// Both per-channel and group-wise 'scale/zp' are supported.

namespace ov::intel_gpu::mlir::decompression {

// Compressed weights and their decompression parameters, all of static shapes. `zero_point` is null
// if there is none. MLIR integers are signless, so the signedness comes from the OV element types.
struct CompressedWeights {
    Value weights;
    bool weights_signed;
    Value scale;
    Value zero_point;
    bool zp_signed = false;
};

inline ArrayRef<int64_t> static_shape(Value v) {
    return mlir::cast<RankedTensorType>(v.getType()).getShape();
}

// Physical order to request for a group-wise parameter [..., N, G] (G > 1): N innermost, i.e. the two
// last dims swapped, so the kernel reads it along N. Nothing is needed for other parameters.
inline std::optional<ConversionContext::DimOrder> grouped_param_order(ArrayRef<int64_t> param_shape) {
    const auto rank = param_shape.size();
    if (rank < 2 || param_shape[rank - 1] <= 1) {
        return std::nullopt;
    }
    ConversionContext::DimOrder order(rank);
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[rank - 1], order[rank - 2]);
    return order;
}

// Indexing map for a scale/zp tensor:
//   weights [E, N, K], scale [E, N, G]:  (e, n, k) -> (e, n, k floordiv (K / G))
inline AffineMap get_scale_zp_indexing_map(MLIRContext* ctx, ArrayRef<int64_t> scale_zp_shape, ArrayRef<int64_t> weights_shape) {
    const auto rank = weights_shape.size();
    OPENVINO_ASSERT(scale_zp_shape.size() <= rank, "Scale / zp rank ", scale_zp_shape.size(), " exceeds the weights rank ", rank);
    // Offset to handle cases where 'scale/zp' have lower rank than weights
    const auto offset = rank - scale_zp_shape.size();
    SmallVector<AffineExpr> exprs;
    for (size_t i = 0; i < scale_zp_shape.size(); ++i) {
        const auto dim = offset + i;
        const auto scale_zp_size = scale_zp_shape[i];
        const auto weights_size = weights_shape[dim];
        if (scale_zp_size == weights_size) {
            exprs.push_back(getAffineDimExpr(dim, ctx));
        } else if (scale_zp_size == 1) {
            // One value shared by the whole dim: always index 0.
            exprs.push_back(getAffineConstantExpr(0, ctx));
        } else {
            // "Grouped" K-dim ('K / G')
            OPENVINO_ASSERT(dim == rank - 1 && weights_size % scale_zp_size == 0,
                            "Scale / zp dim ",
                            scale_zp_size,
                            " doesn't fit the weights dim ",
                            weights_size,
                            " at axis ",
                            dim);
            exprs.push_back(getAffineDimExpr(dim, ctx).floorDiv(weights_size / scale_zp_size));
        }
    }
    return AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, exprs, ctx);
}

// Extends an integer to i32 respecting its signedness (no-op for i32).
inline Value extend_to_i32(OpBuilder& builder, Location loc, Value source_value, bool is_signed) {
    const auto i32 = builder.getI32Type();
    if (source_value.getType() == i32) {
        return source_value;
    }
    if (is_signed) {
        return arith::ExtSIOp::create(builder, loc, i32, source_value);
    }
    return arith::ExtUIOp::create(builder, loc, i32, source_value);
}

// Casts a float to the float `target_type` (no-op if the types already match).
inline Value cast_float(OpBuilder& builder, Location loc, Value source_value, Type target_type) {
    const auto source_type = source_value.getType();
    if (source_type == target_type) {
        return source_value;
    }
    if (source_type.getIntOrFloatBitWidth() < target_type.getIntOrFloatBitWidth()) {
        return arith::ExtFOp::create(builder, loc, target_type, source_value, /*fastmath=*/arith::FastMathFlagsAttr{});
    }
    return arith::TruncFOp::create(builder, loc, target_type, source_value);
}

// Dequantizes the weights into a tensor of the same shape and `ov_out_type` by a single elementwise
// linalg.generic (so it can be fused into the consumer):
//   out = (w - zp) * scale
// Steps:
//   1. Indexing maps: identity for the weights and the output, get_scale_zp_indexing_map() for 'scale/zp'
//   2. The body: subtract an integer zp in i32 and convert to float (a float zp is subtracted after
//      the conversion), then multiply by scale.
inline Value decompress_weights(ConversionContext& context, Location op_loc, const CompressedWeights& cw, const ov::element::Type& ov_out_type) {
    auto& op_builder = context.builder();
    auto* ctx = context.context;
    const auto weights_shape = static_shape(cw.weights);
    const auto rank = weights_shape.size();
    const bool has_zp = static_cast<bool>(cw.zero_point);
    const bool float_zp = has_zp && mlir::isa<FloatType>(getElementTypeOrSelf(cw.zero_point.getType()));

    const auto out_elem_type = importPrecision(ctx, ov_out_type);
    const auto out_type = RankedTensorType::get(weights_shape, out_elem_type);
    Value empty = tensor::EmptyOp::create(op_builder, op_loc, out_type, ValueRange{}).getResult();

    // Inputs of the generic: [weights, scale, (zp)].
    SmallVector<Value> ins{cw.weights, cw.scale};
    if (has_zp) {
        ins.push_back(cw.zero_point);
    }

    // 1. Indexing maps, one per input and one for the output, in the same order.
    const auto identity = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<AffineMap> maps{/*weights_map=*/identity,
                                /*scale_map=*/get_scale_zp_indexing_map(ctx, static_shape(cw.scale), weights_shape)};
    if (has_zp) {
        maps.push_back(/*zp_map=*/get_scale_zp_indexing_map(ctx, static_shape(cw.zero_point), weights_shape));
    }
    maps.push_back(/*output_map=*/identity);

    // 2. Body: out = (w - zp) * scale.
    const SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        op_builder,
        op_loc,
        TypeRange{out_type},
        ins,
        ValueRange{empty},
        maps,
        iterators,
        [&](OpBuilder& builder, Location loc, ValueRange args) {
            const Value compressed_weight = args[0];
            const Value scale = args[1];
            const Value zero_point = has_zp ? args[2] : Value{};

            Value weight_i32 = extend_to_i32(builder, loc, compressed_weight, cw.weights_signed);
            if (has_zp && !float_zp) {
                weight_i32 = arith::SubIOp::create(builder, loc, weight_i32, extend_to_i32(builder, loc, zero_point, cw.zp_signed));
            }
            Value dequantized_weight = arith::SIToFPOp::create(builder, loc, out_elem_type, weight_i32);
            if (float_zp) {
                dequantized_weight = arith::SubFOp::create(builder, loc, dequantized_weight, cast_float(builder, loc, zero_point, out_elem_type));
            }
            dequantized_weight = arith::MulFOp::create(builder, loc, dequantized_weight, cast_float(builder, loc, scale, out_elem_type));
            linalg::YieldOp::create(builder, loc, dequantized_weight);
        });
    return generic.getResult(0);
}

}  // namespace ov::intel_gpu::mlir::decompression
