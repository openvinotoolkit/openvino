// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patterns.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/floor.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/reduce_prod.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../common/converters/relu.hpp"
#include "../common/converters/concat.hpp"
#include "../common/converters/floor.hpp"
#include "../common/converters/gather.hpp"
#include "../common/converters/matmul.hpp"
#include "../common/converters/reduce.hpp"
#include "../common/converters/sdpa.hpp"
#include "../common/converters/shape_of.hpp"
#include "../common/converters/slice.hpp"
#include "../common/converters/squeeze.hpp"
#include "../common/converters/transpose.hpp"
#include "../common/converters/unsqueeze.hpp"
#include "../common/converters/binary_eltwise.hpp"

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

ReluPattern::ReluPattern()
    : MarkPattern(wrap_type<v0::Relu>({any_input()}, elementwise_no_broadcast_predicate), ConvertRelu()) {}

ConcatPattern::ConcatPattern()
    : MarkPattern(wrap_type<v0::Concat>(), ConvertConcat()) {}

FloorPattern::FloorPattern()
    : MarkPattern(wrap_type<v0::Floor>({any_input()}), ConvertFloor()) {}

GatherPattern::GatherPattern()
    : MarkPattern(wrap_type<v8::Gather>({any_input(), any_input(), any_input()}), ConvertGather()) {}

MatMulPattern::MatMulPattern()
    : MarkPattern(
        wrap_type<v0::MatMul>({any_input(), any_input()}, [](const Output<Node>& output) {
            auto node = std::dynamic_pointer_cast<v0::MatMul>(output.get_node_shared_ptr());
            assert(node);
            // FIXME: current code limitation
            return !has_dynamic_rank(node) && !(node->get_transpose_a() && node->get_transpose_b()) &&
                   node->get_input_partial_shape(0).rank().get_length() == 2 &&
                   node->get_input_partial_shape(1).rank().get_length() == 2;
        }),
        ConvertMatMul()) {}

template <typename OVOp>
ReducePattern<OVOp>::ReducePattern()
    : MarkPattern(std::make_shared<pass::pattern::op::WrapType>(OVOp::get_type_info_static()), ConvertReduce<OVOp>()) {}

// Explicit template instantiations
template class ReducePattern<ov::op::v1::ReduceMax>;
template class ReducePattern<ov::op::v1::ReduceMean>;
template class ReducePattern<ov::op::v1::ReduceMin>;
template class ReducePattern<ov::op::v1::ReduceProd>;
template class ReducePattern<ov::op::v1::ReduceSum>;

SDPAPattern::SDPAPattern()
    : MarkPattern(wrap_type<v13::ScaledDotProductAttention>(), ConvertSDPA()) {}

ShapeOfPattern::ShapeOfPattern()
    : MarkPattern(wrap_type<v3::ShapeOf>({any_input()}), ConvertShapeOf()) {}

SlicePattern::SlicePattern()
    : MarkPattern(wrap_type<v8::Slice>({any_input(), any_input(), any_input(), any_input(), any_input()}), ConvertSlice()) {}

SqueezePattern::SqueezePattern()
    : MarkPattern(wrap_type<v0::Squeeze>({any_input()}), ConvertSqueeze()) {}

TransposePattern::TransposePattern()
    : MarkPattern(wrap_type<v1::Transpose>({any_input(), any_input()}), ConvertTranspose()) {}

UnsqueezePattern::UnsqueezePattern()
    : MarkPattern(wrap_type<v0::Unsqueeze>({any_input(), any_input()}), ConvertUnsqueeze()) {}

BinaryEltwisePatternBase::BinaryEltwisePatternBase(
    NodeTypeInfo wrapped_type, GraphConverter::Convertor convertor, const std::set<element::Type>& element_types)
    : MarkPattern(
        std::make_shared<pass::pattern::op::WrapType>(
            wrapped_type,
            [element_types](const Output<Node>& output) {
                if (!element_types.empty() && !element_types.count(output.get_element_type())) {
                    return false;
                }
                auto node = output.get_node_shared_ptr();
                for (const auto& input : node->inputs()) {
                    if (!statically_broadcastable(input.get_partial_shape(), output.get_partial_shape())) {
                        return false;
                    }
                }
                return true;
            },
            OutputVector{any_input(), any_input()}),
        convertor) {}

template <typename OVOp, typename LinalgOp>
BinaryEltwisePattern<OVOp, LinalgOp>::BinaryEltwisePattern(const std::set<element::Type>& element_types)
    : BinaryEltwisePatternBase(OVOp::get_type_info_static(), ConvertBinaryEltwise<LinalgOp>(), element_types) {}

// Explicit template instantiations
template class BinaryEltwisePattern<v1::Add, linalg::AddOp>;
template class BinaryEltwisePattern<v1::Subtract, linalg::SubOp>;
template class BinaryEltwisePattern<v1::Multiply, linalg::MulOp>;
template class BinaryEltwisePattern<v1::Divide, linalg::DivOp>;

}  // namespace mlir
}  // namespace ov
