// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/relu.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "relu.hpp"
#include "../convert_common.hpp"

namespace {

using namespace ov::mlir;

struct ConvertRelu {
    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        auto dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);
        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamic_dimensions);
        auto zero = getConstant(builder, ov_output_element_type, 0);
        auto fill = builder.create<linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});
        auto relu =
            builder.create<linalg::MaxOp>(loc, mlir::ValueRange{input, fill.getResult(0)}, mlir::ValueRange{empty});
        context.addOutputs(node, relu);
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

ReluPattern::ReluPattern()
    : MarkPattern(wrap_type<v0::Relu>({any_input()}, elementwise_no_broadcast_predicate), ConvertRelu()) {}

}  // namespace mlir
}  // namespace ov
