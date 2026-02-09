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
                                                               bool hasMask) {
        AffineExpr m, n, k1, k2;
        bindDims(ctx, m, n, k1, k2);

        auto qMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k1}, ctx);
        auto kMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, k1}, ctx);
        auto vMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, n}, ctx);
        auto sMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, ctx);
        auto rMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, n}, ctx);
        if (hasMask) {
            // Add mask map only if it exists
            auto mMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k2}, ctx);
            return {qMap, kMap, vMap, sMap, mMap, rMap};
        }
        return {qMap, kMap, vMap, sMap, rMap};
    }

    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        // TODO: Support broadcasts
        const auto inputs = context.getInputs(node);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamic_dimensions);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = builder.create<linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});

        mlir::SmallVector<Value, 3> ins{inputs[0], inputs[1], inputs[2]};
        mlir::SmallVector<Value, 1> outs{fill.getResult(0)};

        auto matmul_node = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(node);
        assert(matmul_node);

        Operation* sdpa;
        SmallVector<AffineMap> indexingMaps =
            getStandardAttentionIndexingMaps(context.context, false);
        // FIXME: extract actual scale
        Value scale = getConstant(builder, ov_output_element_type, 1.0f);
        sdpa = builder.create<linalgx::AttentionOp>(
            loc, outs[0].getType(), inputs[0], inputs[1], inputs[2], scale, outs[0],
            builder.getAffineMapArrayAttr(indexingMaps), /*mask=*/nullptr);
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
