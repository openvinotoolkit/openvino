// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/matmul.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "matmul.hpp"
#include "../convert_common.hpp"


namespace {

using namespace ov::mlir;

struct ConvertMatMul {
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
        // TODO: Add other variants of transpose_a/transpose_b
        auto matmul = builder.create<linalg::MatmulTransposeBOp>(loc, mlir::ValueRange{inputs[0], inputs[1]}, mlir::ValueRange{fill.getResult(0)});
        context.addOutputs(node, matmul);
    }
};

}

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

MatMulPattern::MatMulPattern() : MarkPattern(
    wrap_type<v0::MatMul>({any_input(), any_input()}, [](const Output<Node>& output) {
        auto node = std::dynamic_pointer_cast<v0::MatMul>(output.get_node_shared_ptr());
        assert(node);
        // FIXME: current code limitation
        return
            !has_dynamic_rank(node) &&
            !node->get_transpose_a() && node->get_transpose_b() &&
            node->get_input_partial_shape(0).rank().get_length() == 2 &&
            node->get_input_partial_shape(1).rank().get_length() == 2;
    }),
    ConvertMatMul()) {
    }


} // namespace mlir
} // namespace ov