// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patterns.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/floor.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/reduce_prod.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/ceiling.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/tanh.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <ov_ops/rms.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../common/converters/relu.hpp"
#include "../common/converters/unary_eltwise.hpp"
#include "../common/converters/concat.hpp"
#include "../common/converters/floor.hpp"
#include "../common/converters/gather.hpp"
#include "../common/converters/matmul.hpp"
#include "../common/converters/reduce.hpp"
#include "../common/converters/reshape.hpp"
#include "../common/converters/rms.hpp"
#include "../common/converters/sdpa.hpp"
#include "../common/converters/shape_of.hpp"
#include "../common/converters/slice.hpp"
#include "../common/converters/squeeze.hpp"
#include "../common/converters/transpose.hpp"
#include "../common/converters/unsqueeze.hpp"
#include "../common/converters/binary_eltwise.hpp"

namespace ov::intel_gpu::mlir {

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
            return !has_dynamic_rank(node) && !(node->get_transpose_a() && node->get_transpose_b());
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

RMSPattern::RMSPattern()
    : MarkPattern(
        wrap_type<ov::op::internal::RMS>([](const Output<Node>& output) {
            auto node = ov::as_type_ptr<ov::op::internal::RMS>(output.get_node_shared_ptr());
            if (!node || has_dynamic_rank(node) || !output.get_element_type().is_real()) {
                return false;
            }
            // The converter computes the mean over the last dimension, so it must be static.
            const auto shape = output.get_partial_shape();
            if (shape[shape.rank().get_length() - 1].is_dynamic()) {
                return false;
            }
            // Mixed input/output precision (RMS output_type attribute) is not supported
            if (node->get_input_element_type(0) != output.get_element_type()) {
                return false;
            }
            return !node->get_elementwise_affine() ||
                   (node->get_input_element_type(1) == output.get_element_type() &&
                    statically_broadcastable(node->get_input_partial_shape(1), shape));
        }),
        ConvertRMS()) {}

ReshapePattern::ReshapePattern()
    : MarkPattern(wrap_type<v1::Reshape>({any_input(), any_input()}), ConvertReshape()) {}

SDPAPattern::SDPAPattern()
    : MarkPattern(
        wrap_type<v13::ScaledDotProductAttention>([](const Output<Node>& output) {
            auto node = std::dynamic_pointer_cast<v13::ScaledDotProductAttention>(output.get_node_shared_ptr());
            if (!node) {
                return false;
            }

            // Query, Key, Value ranks must be static, equal, and either 3D or 4D
            const auto q_shape = node->get_input_partial_shape(0);
            const auto k_shape = node->get_input_partial_shape(1);
            const auto v_shape = node->get_input_partial_shape(2);
            if (q_shape.rank().is_dynamic() || k_shape.rank().is_dynamic() || v_shape.rank().is_dynamic()) {
                return false;
            }
            const auto q_rank = q_shape.rank().get_length();
            if (q_rank != k_shape.rank().get_length() || q_rank != v_shape.rank().get_length()) {
                return false;
            }
            if (q_rank != 3 && q_rank != 4) {
                return false;
            }

            // Causal attention is not supported
            if (node->get_causal()) {
                return false;
            }

            const auto input_size = node->get_input_size();
            // Sink parameter (6th input) is not supported
            if (input_size >= 6) {
                return false;
            }

            // Mask (input 3): only static shapes are supported, dynamic ones are rejected
            if (input_size > 3) {
                const auto mask_shape = node->get_input_partial_shape(3);
                const bool has_mask = mask_shape.rank().is_dynamic() || mask_shape.rank().get_length() > 0;
                if (has_mask && mask_shape.is_dynamic()) {
                    return false;
                }
            }

            // Scale (input 4) must be a Constant, dynamic scale input is not supported
            if (input_size > 4 &&
                !std::dynamic_pointer_cast<v0::Constant>(node->get_input_node_shared_ptr(4))) {
                return false;
            }

            return true;
        }),
        ConvertSDPA()) {}

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
template class BinaryEltwisePattern<v1::Power, linalg::PowFOp>;

template <typename OVOp, typename LinalgOp>
UnaryEltwisePattern<OVOp, LinalgOp>::UnaryEltwisePattern()
    : MarkPattern(wrap_type<OVOp>({any_input()}), ConvertUnaryEltwise<LinalgOp>()) {}

// Explicit template instantiations
template class UnaryEltwisePattern<v0::Abs, linalg::AbsOp>;
template class UnaryEltwisePattern<v0::Ceiling, linalg::CeilOp>;
template class UnaryEltwisePattern<v0::Exp, linalg::ExpOp>;
template class UnaryEltwisePattern<v0::Log, linalg::LogOp>;
template class UnaryEltwisePattern<v0::Negative, linalg::NegFOp>;
template class UnaryEltwisePattern<v0::Sqrt, linalg::SqrtOp>;
template class UnaryEltwisePattern<v0::Tanh, linalg::TanhOp>;

}  // namespace ov::intel_gpu::mlir
