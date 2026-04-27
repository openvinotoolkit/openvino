// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/constant.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/reduce_prod.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/util/arithmetic_reductions_keep_dims.hpp>

#include "../convert_common.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace mlir {

using Value = ::mlir::Value;
using ValueRange = ::mlir::ValueRange;

template <typename OVOp>
struct ConvertReduce {
    ConvertReduce() = default;

    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto el_type = importPrecision(context.context, node->get_input_element_type(0));
        auto input_shape = node->get_input_partial_shape(0);
        auto input_rank = input_shape.rank().get_length();
        SmallVector<int64_t> reduction_axes;
        {
            auto input1 = dynamic_cast<ov::op::v0::Constant*>(node->get_input_node_ptr(1));
            assert(input1 && "Only constant axes are supported");
            auto axes = input1->cast_vector<int64_t>();
            reduction_axes.reserve(axes.size());
            for (int64_t axis : axes) {
                reduction_axes.push_back(axis < 0 ? axis + input_rank : axis);
            }
        }
        ::mlir::RankedTensorType result_type;
        {
            SmallVector<int64_t> shape;
            for (size_t i = 0; i < input_rank; ++i) {
                if (!llvm::is_contained(reduction_axes, i)) {
                    auto dim = input_shape[i];
                    assert(dim.is_static() && "Dynamic shapes not supported");
                    shape.push_back(dim.get_length());
                }
            }
            result_type = RankedTensorType::get(shape, el_type);
        }

        auto& builder = context.builder();
        auto loc = createLocation(context.context, node);
        auto empty = ::mlir::tensor::EmptyOp::create(builder, loc, result_type, ValueRange{});
        Value init_value = create_init_value(builder, loc, el_type);
        auto output = ::mlir::linalg::FillOp::create(builder, loc, ValueRange{init_value}, ValueRange{empty});
        Value result = ::mlir::linalg::ReduceOp::create(
                           builder,
                           loc,
                           ValueRange{context.getInputs(node)[0]},
                           ValueRange{output.getResult(0)},
                           reduction_axes,
                           [&](::mlir::OpBuilder& b, ::mlir::Location loc, ValueRange inputs) {
                               Value result = create_payload_op(b, loc, inputs[0], inputs[1], el_type);
                               ::mlir::linalg::YieldOp::create(b, loc, result);
                           })
                           .getResult(0);

        // For ReduceMean, divide by the number of elements
        if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceMean>) {
            // Calculate the number of elements in the reduction dimensions
            int64_t num_els = 1;
            for (auto axis : reduction_axes) {
                num_els *= input_shape[axis].get_length();
            }
            ::mlir::TypedAttr divisor_attr;
            if (el_type.isInteger()) {
                divisor_attr = builder.getIntegerAttr(el_type, num_els);
            } else {
                divisor_attr = builder.getFloatAttr(el_type, static_cast<double>(num_els));
            }
            auto divisor = ::mlir::arith::ConstantOp::create(builder,
                                                             loc,
                                                             ::mlir::DenseElementsAttr::get(result_type, divisor_attr));
            result = ::mlir::linalg::DivOp::create(builder, loc, ValueRange{result, divisor}, ValueRange{result})
                         .getResult(0);
        }

        // If keep_dims is true, broadcast along the reduced dimensions
        if (auto keep_dims = dynamic_cast<const ov::op::util::ArithmeticReductionKeepDims*>(node.get());
            keep_dims && keep_dims->get_keep_dims()) {
            auto shape = llvm::map_to_vector(node->get_output_partial_shape(0), [](const ov::Dimension& dim) {
                return dim.get_length();
            });
            auto empty = ::mlir::tensor::EmptyOp::create(builder, loc, shape, el_type);
            result = ::mlir::linalg::BroadcastOp::create(builder, loc, result, empty, reduction_axes).getResult()[0];
        }

        return result.getDefiningOp();
    }

private:
    Value create_init_value(::mlir::OpBuilder& builder, ::mlir::Location loc, ::mlir::Type type) {
        if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceMax>) {
            if (type.isFloat()) {
                return getConstant(builder, type, -std::numeric_limits<double>::infinity(), loc);
            } else {
                int64_t min_val = type.isUnsignedInteger() ? 0 : -(1LL << (type.getIntOrFloatBitWidth() - 1));
                return getConstant(builder, type, min_val, loc);
            }
        } else if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceMin>) {
            if (type.isFloat()) {
                return getConstant(builder, type, std::numeric_limits<double>::infinity(), loc);
            } else {
                unsigned bitwidth = type.getIntOrFloatBitWidth();
                int64_t max_val = type.isUnsignedInteger() ? ((1ULL << bitwidth) - 1) : ((1LL << (bitwidth - 1)) - 1);
                return getConstant(builder, type, max_val, loc);
            }
        } else if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceSum> ||
                             std::is_same_v<OVOp, ov::op::v1::ReduceMean>) {
            return getConstant(builder, type, 0, loc);
        } else if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceProd>) {
            return getConstant(builder, type, 1, loc);
        } else {
            static_assert(sizeof(OVOp) == 0, "Unsupported reduction operation");
        }
    }

    Value create_payload_op(::mlir::OpBuilder& builder, ::mlir::Location loc, Value lhs, Value rhs, ::mlir::Type type) {
        if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceMax>) {
            if (type.isFloat()) {
                return ::mlir::arith::MaximumFOp::create(builder, loc, lhs, rhs);
            } else if (type.isUnsignedInteger()) {
                return ::mlir::arith::MaxUIOp::create(builder, loc, lhs, rhs);
            } else {
                return ::mlir::arith::MaxSIOp::create(builder, loc, lhs, rhs);
            }
        } else if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceMin>) {
            if (type.isFloat()) {
                return ::mlir::arith::MinimumFOp::create(builder, loc, lhs, rhs);
            } else if (type.isUnsignedInteger()) {
                return ::mlir::arith::MinUIOp::create(builder, loc, lhs, rhs);
            } else {
                return ::mlir::arith::MinSIOp::create(builder, loc, lhs, rhs);
            }
        } else if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceSum> ||
                             std::is_same_v<OVOp, ov::op::v1::ReduceMean>) {
            if (type.isFloat()) {
                return ::mlir::arith::AddFOp::create(builder, loc, lhs, rhs);
            } else {
                return ::mlir::arith::AddIOp::create(builder, loc, lhs, rhs);
            }
        } else if constexpr (std::is_same_v<OVOp, ov::op::v1::ReduceProd>) {
            if (type.isFloat()) {
                return ::mlir::arith::MulFOp::create(builder, loc, lhs, rhs);
            } else {
                return ::mlir::arith::MulIOp::create(builder, loc, lhs, rhs);
            }
        } else {
            static_assert(sizeof(OVOp) == 0, "Unsupported reduction operation");
        }
    }
};

}  // namespace mlir
}  // namespace ov
