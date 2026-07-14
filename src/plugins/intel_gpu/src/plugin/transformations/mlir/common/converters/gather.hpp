// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include <openvino/op/gather.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"

namespace ov {
namespace mlir {

struct ConvertGather {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        // TODO: support batch attribute
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        const auto indices = context.getInputs(node)[1];
        // get_axis() seems to be enough?
        // const auto axis = context.getInputs(node)[2];
        
        const auto ov_index_element_type = node->get_input_element_type(1);
        const auto ov_index_shape = node->get_input_partial_shape(1);
        auto dynamic_index_dims = context.get_dynamic_dimension_values(ov_index_shape);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto out_type = importTensor(context.context, ov_output_shape, ov_output_element_type);
        RankedTensorType indices_type = importTensor(context.context, ov_index_shape, ov_index_element_type);

        bool is_input_scalar = ov_index_shape.rank().is_static() && ov_index_shape.rank().get_length() == 0;
        
        // `shape_of` returns tensor<1xindex> for scalars, gather requires dimention match at `gather_dims` indices.
        // This expands the shape of input indices to be <1xi64> in case of scalars to resolve types mismatch.
        Value indices_expanded = indices;
        if (is_input_scalar) {
            SmallVector<int64_t> new_shape({1});
            indices_type = RankedTensorType::get(new_shape, importPrecision(context.context, ov_index_element_type));
            SmallVector<ReassociationIndices> reassociation; // intentionally empty for scalar
            auto expanded = tensor::ExpandShapeOp::create(builder, loc, indices_type, indices, reassociation);
            indices_expanded = expanded.getResult();
        }

        // Convert negative indices into positive ones: compare to zero and select from orinal or a sum based on
        // the resulting predicate.
        auto empty = tensor::EmptyOp::create(builder, loc, indices_type, dynamic_index_dims);
        auto zero = getConstant(builder, ov_index_element_type, 0);
        auto fill = linalg::FillOp::create(builder, loc, mlir::ValueRange{zero}, mlir::ValueRange{empty});
        auto pred = arith::CmpIPredicate::slt;
        auto cmpi = arith::CmpIOp::create(builder, loc, pred, indices_expanded, fill.getResult(0));
        auto shape_of = shape::ShapeOfOp::create(builder, loc, mlir::ValueRange{input});
        auto cast = arith::IndexCastOp::create(builder, loc, indices_type, mlir::ValueRange{shape_of});

        auto empty_add = tensor::EmptyOp::create(builder, loc, indices_expanded.getType(), dynamic_index_dims);
        auto add = linalg::AddOp::create(builder, loc, mlir::ValueRange{cast.getResult(), indices_expanded}, mlir::ValueRange{empty_add});
        auto select = linalg::SelectOp::create(builder, loc, mlir::ValueRange{cmpi.getResult(), add.getResult(0), indices_expanded}, mlir::ValueRange{empty_add});

        auto gather_node = std::dynamic_pointer_cast<ov::op::util::GatherBase>(node);
        assert(gather_node && "Expected a gather node");
        int64_t axis = gather_node->get_axis();

        llvm::SmallVector<int64_t> gather_dims{axis};
        auto gather = tensor::GatherOp::create(builder, loc, out_type, input, select.getResult(0), gather_dims, false);
        return gather;
    }
};

}  // namespace mlir
}  // namespace ov

