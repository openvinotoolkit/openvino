// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/scaled_dot_product_attention.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "mlir/IR/AffineExpr.h"

#include "sdpa.hpp"
#include "../convert_common.hpp"

namespace {

using namespace ov::mlir;

struct ConvertSDPA {
    static SmallVector<AffineMap> getStandardAttentionIndexingMaps(MLIRContext *ctx,
                                                               bool hasMask,
                                                               int rank) {
        if (rank == 3) {
            // 3D: (batch, seq, hidden)
            AffineExpr batch, m, k1, k2, n;
            bindDims(ctx, batch, m, k1, k2, n);

            auto qMap = AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, m, k1}, ctx);
            auto kMap = AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, k2, k1}, ctx);
            auto vMap = AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, k2, n}, ctx);
            auto sMap = AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, ctx);
            auto rMap = AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, m, n}, ctx);
            if (hasMask) {
                auto mMap = AffineMap::get(/*dimCount=*/5, /*symbolCount=*/0, {batch, m, k2}, ctx);
                return {qMap, kMap, vMap, sMap, mMap, rMap};
            }
            return {qMap, kMap, vMap, sMap, rMap};
        } else if (rank == 4) {
            // 4D: (batch, head, seq, hidden)
            AffineExpr batch, head, m, k1, k2, n;
            bindDims(ctx, batch, head, m, k1, k2, n);

            auto qMap = AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, {batch, head, m, k1}, ctx);
            auto kMap = AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, {batch, head, k2, k1}, ctx);
            auto vMap = AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, {batch, head, k2, n}, ctx);
            auto sMap = AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, ctx);
            auto rMap = AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, {batch, head, m, n}, ctx);
            if (hasMask) {
                auto mMap = AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, {batch, head, m, k2}, ctx);
                return {qMap, kMap, vMap, sMap, mMap, rMap};
            }
            return {qMap, kMap, vMap, sMap, rMap};
        }
        // Should never reach here due to validation
        return {};
    }

    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);

        // Validate input ranks
        auto qShape = node->get_input_partial_shape(0);
        auto kShape = node->get_input_partial_shape(1);
        auto vShape = node->get_input_partial_shape(2);

        auto qRank = qShape.rank().get_length();
        auto kRank = kShape.rank().get_length();
        auto vRank = vShape.rank().get_length();

        OPENVINO_ASSERT(qRank == kRank && qRank == vRank,
                        "SDPA: Query, Key, and Value must have equal ranks, but got Q rank=",
                        qRank,
                        ", K rank=",
                        kRank,
                        ", V rank=",
                        vRank);

        OPENVINO_ASSERT(qRank == 3 || qRank == 4,
                        "SDPA: Only 3D and 4D inputs are supported, but got rank=",
                        qRank);

        auto sdpa_node = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(node);
        OPENVINO_ASSERT(sdpa_node, "Failed to cast to ScaledDotProductAttention");

        const auto input_size = node->get_input_size();
        const bool causal = sdpa_node->get_causal();
        OPENVINO_ASSERT(!causal, "SDPA: Causal attention is not supported in this version");
        OPENVINO_ASSERT(input_size < 6, "SDPA: sink parameter is not supported");

        const auto ov_output_element_type = node->get_output_element_type(0);

        // Extract mask (input 3) if present and not causal
        Value mask = nullptr;
        bool hasMask = false;
        if (input_size > 3 && !causal && node->get_input_partial_shape(3).rank().get_length() > 0) {
            mask = inputs[3];
            hasMask = true;
        }

        // Extract or compute scale (input 4)
        Value scale;
        if (input_size > 4) {
            // Scale is provided as input tensor - extract scalar from 0-dimensional tensor
            // For a scalar tensor (shape {}), tensor.extract with no indices extracts the value
            scale = builder.create<tensor::ExtractOp>(loc, inputs[4], mlir::ValueRange{});
        } else {
            // Default scale: 1.0
            scale = getConstant(builder, ov_output_element_type, 1.0);
        }

        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamic_dimensions);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = builder.create<linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});

        SmallVector<AffineMap> indexingMaps =
            getStandardAttentionIndexingMaps(context.context, hasMask, qRank);

        Operation* sdpa = builder.create<linalgx::AttentionOp>(
            loc, fill.getResult(0).getType(), inputs[0], inputs[1], inputs[2], scale, fill.getResult(0),
            builder.getAffineMapArrayAttr(indexingMaps), mask);
        context.addOutputs(node, sdpa);
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

SDPAPattern::SDPAPattern()
    : MarkPattern(wrap_type<v13::ScaledDotProductAttention>(), ConvertSDPA()) {}

}  // namespace mlir
}  // namespace ov
