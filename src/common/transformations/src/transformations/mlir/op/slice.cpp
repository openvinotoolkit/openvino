// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/slice.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "slice.hpp"
#include "../convert_common.hpp"

namespace {

using namespace ov::mlir;

struct ConvertSlice {
    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto start = context.getInputs(node)[1];
        const auto stop = context.getInputs(node)[2];
        const auto step = context.getInputs(node)[3];
        const auto axes = context.getInputs(node)[4];

        const auto ov_index_shape = node->get_input_partial_shape(1);
        const auto ov_index_element_type = node->get_input_element_type(1);
        auto dynamic_index_dims = context.get_dynamic_dimension_values(ov_index_shape);

        auto index_type = importTensor(context.context, ov_index_shape, ov_index_element_type);
        auto empty = builder.create<tensor::EmptyOp>(loc, index_type, dynamic_index_dims);

        // TODO: this only works for the all-positive numbers case.
        auto sizes = builder.create<linalg::SubOp>(loc, mlir::ValueRange{stop, start}, mlir::ValueRange{empty});
        auto slice = builder.create<tensor::ExtractSliceOp>(loc, input, mlir::ValueRange{start}, mlir::ValueRange{sizes.getResults()}, mlir::ValueRange{step});
        context.addOutputs(node, slice);
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

SlicePattern::SlicePattern() : MarkPattern(wrap_type<v8::Slice>({any_input(), any_input(), any_input(), any_input(), any_input()}), ConvertSlice()) {}

}  // namespace mlir
}  // namespace ov
