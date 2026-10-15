// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patterns.hpp"

#include <algorithm>
#include <functional>
#include <openvino/op/abs.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/ceiling.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/floor.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/reduce_prod.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/tanh.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <ov_ops/rms.hpp>

#include "../common/converters/binary_eltwise.hpp"
#include "../common/converters/concat.hpp"
#include "../common/converters/floor.hpp"
#include "../common/converters/gather.hpp"
#include "../common/converters/matmul.hpp"
#include "../common/converters/reduce.hpp"
#include "../common/converters/relu.hpp"
#include "../common/converters/reshape.hpp"
#include "../common/converters/rms.hpp"
#include "../common/converters/sdpa.hpp"
#include "../common/converters/shape_of.hpp"
#include "../common/converters/slice.hpp"
#include "../common/converters/squeeze.hpp"
#include "../common/converters/transpose.hpp"
#include "../common/converters/unary_eltwise.hpp"
#include "../common/converters/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu::mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

ReluPattern::ReluPattern() : MarkPattern(wrap_type<v0::Relu>({any_input()}, elementwise_no_broadcast_predicate), ConvertRelu()) {}

ConcatPattern::ConcatPattern() : MarkPattern(wrap_type<v0::Concat>(), ConvertConcat()) {}

FloorPattern::FloorPattern() : MarkPattern(wrap_type<v0::Floor>({any_input()}), ConvertFloor()) {}

GatherPattern::GatherPattern() : MarkPattern(wrap_type<v8::Gather>({any_input(), any_input(), any_input()}), ConvertGather()) {}

MatMulPattern::MatMulPattern()
    : MarkPattern(wrap_type<v0::MatMul>({any_input(), any_input()},
                                        [](const Output<Node>& output) {
                                            auto node = std::dynamic_pointer_cast<v0::MatMul>(output.get_node_shared_ptr());
                                            assert(node);
                                            return !has_dynamic_rank(node) && (!node->get_transpose_a() || !node->get_transpose_b());
                                        }),
                  ConvertMatMul()) {}

template <typename OVOp>
ReducePattern<OVOp>::ReducePattern() : MarkPattern(std::make_shared<pass::pattern::op::WrapType>(OVOp::get_type_info_static()), ConvertReduce<OVOp>()) {}

// Explicit template instantiations
template class ReducePattern<ov::op::v1::ReduceMax>;
template class ReducePattern<ov::op::v1::ReduceMean>;
template class ReducePattern<ov::op::v1::ReduceMin>;
template class ReducePattern<ov::op::v1::ReduceProd>;
template class ReducePattern<ov::op::v1::ReduceSum>;

RMSPattern::RMSPattern()
    : MarkPattern(wrap_type<ov::op::internal::RMS>([](const Output<Node>& output) {
                      auto node = ov::as_type_ptr<ov::op::internal::RMS>(output.get_node_shared_ptr());
                      if (!node || has_dynamic_rank(node) || !output.get_element_type().is_real()) {
                          return false;
                      }
                      // The converter computes the mean over the last dimension, so it must be static.
                      const auto& shape = output.get_partial_shape();
                      if (shape[shape.rank().get_length() - 1].is_dynamic()) {
                          return false;
                      }
                      // Mixed input/output precision (RMS output_type attribute) is not supported
                      if (node->get_input_element_type(0) != output.get_element_type()) {
                          return false;
                      }
                      return !node->get_elementwise_affine() || (node->get_input_element_type(1) == output.get_element_type() &&
                                                                 statically_broadcastable(node->get_input_partial_shape(1), shape));
                  }),
                  ConvertRMS()) {}

ReshapePattern::ReshapePattern() : MarkPattern(wrap_type<v1::Reshape>({any_input(), any_input()}), ConvertReshape()) {}

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
                  OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — dynamic Q/K/V rank");
                  return false;
              }
              const auto q_rank = q_shape.rank().get_length();
              if (q_rank != k_shape.rank().get_length() || q_rank != v_shape.rank().get_length()) {
                  OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — Q/K/V rank mismatch");
                  return false;
              }
              if (q_rank != 3 && q_rank != 4) {
                  OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — unsupported rank " << q_rank << " (expected 3 or 4)");
                  return false;
              }

              // The attention tiling fully unrolls the head size dimension, so a large head size
              // multiplies kernel size and register pressure. Past this limit the generated kernel
              // spills heavily and loses to the native SDPA implementations both in compile time and
              // in execution time, so leave those nodes alone.
              constexpr int64_t max_head_size = 128;
              const auto& q_head_size = q_shape[q_rank - 1];
              const auto& v_head_size = v_shape[q_rank - 1];
              if (q_head_size.get_min_length() > max_head_size || v_head_size.get_min_length() > max_head_size) {
                  OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — head size " << q_head_size << "/" << v_head_size
                                                                     << " exceeds the limit of " << max_head_size);
                  return false;
              }

              // Causal attention is not supported
              if (node->get_causal()) {
                  OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — causal attention");
                  return false;
              }

              const auto input_size = node->get_input_size();
              // Sink parameter (6th input) is not supported
              if (input_size >= 6) {
                  OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — sink parameter (6th input) is not supported");
                  return false;
              }

              // Mask (input 3): only static shapes are supported, dynamic ones are rejected
              if (input_size > 3) {
                  const auto mask_shape = node->get_input_partial_shape(3);
                  const bool has_mask = mask_shape.rank().is_dynamic() || mask_shape.rank().get_length() > 0;
                  if (has_mask && mask_shape.is_dynamic()) {
                      OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — dynamic mask shape");
                      return false;
                  }
              }

              // Scale (input 4) must be a Constant, dynamic scale input is not supported
              if (input_size > 4 && !std::dynamic_pointer_cast<v0::Constant>(node->get_input_node_shared_ptr(4))) {
                  OPENVINO_MLIR_DEBUG_PRINT("SDPAPattern: rejected " << node->get_friendly_name() << " — non-constant scale input");
                  return false;
              }

              return true;
          }),
          ConvertSDPA()) {}

ShapeOfPattern::ShapeOfPattern() : MarkPattern(wrap_type<v3::ShapeOf>({any_input()}), ConvertShapeOf()) {}

// ConvertSlice only implements the static, all-positive, unit-step case.
SlicePattern::SlicePattern()
    : MarkPattern(wrap_type<v8::Slice>({any_input(), any_input(), any_input(), any_input(), any_input()},
                                       [](const Output<Node>& output) {
                                           auto node = ov::as_type_ptr<v8::Slice>(output.get_node_shared_ptr());
                                           if (!node || has_dynamic_rank(node)) {
                                               OPENVINO_MLIR_DEBUG_PRINT("SlicePattern: rejected " << node->get_friendly_name() << " — dynamic rank");
                                               return false;
                                           }

                                           const auto start = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(1));
                                           const auto stop = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(2));
                                           const auto step = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(3));
                                           if (!start || !stop || !step) {
                                               OPENVINO_MLIR_DEBUG_PRINT("SlicePattern: rejected " << node->get_friendly_name()
                                                                                                   << " — start/stop/step must be Constants");
                                               return false;
                                           }

                                           // Only unit step is supported (sizes = stop - start assumes step == 1).
                                           const auto step_values = step->cast_vector<int64_t>();
                                           if (std::any_of(step_values.begin(), step_values.end(), [](int64_t s) {
                                                   return s != 1;
                                               })) {
                                               OPENVINO_MLIR_DEBUG_PRINT("SlicePattern: rejected " << node->get_friendly_name() << " — non-unit step");
                                               return false;
                                           }

                                           // Negative start/stop are not handled (no bounds normalization in the converter).
                                           const auto start_values = start->cast_vector<int64_t>();
                                           const auto stop_values = stop->cast_vector<int64_t>();
                                           if (std::any_of(start_values.begin(),
                                                           start_values.end(),
                                                           [](int64_t v) {
                                                               return v < 0;
                                                           }) ||
                                               std::any_of(stop_values.begin(), stop_values.end(), [](int64_t v) {
                                                   return v < 0;
                                               })) {
                                               OPENVINO_MLIR_DEBUG_PRINT("SlicePattern: rejected " << node->get_friendly_name() << " — negative start/stop");
                                               return false;
                                           }
                                           return true;
                                       }),
                  ConvertSlice()) {}

SqueezePattern::SqueezePattern() : MarkPattern(wrap_type<v0::Squeeze>({any_input()}), ConvertSqueeze()) {}

// ConvertTranspose requires an explicit Constant order covering all input dimensions.
TransposePattern::TransposePattern()
    : MarkPattern(wrap_type<v1::Transpose>(
                      {any_input(), any_input()},
                      [](const Output<Node>& output) {
                          auto node = ov::as_type_ptr<v1::Transpose>(output.get_node_shared_ptr());
                          if (!node || has_dynamic_rank(node)) {
                              OPENVINO_MLIR_DEBUG_PRINT("TransposePattern: rejected " << output.get_node()->get_friendly_name() << " — dynamic rank");
                              return false;
                          }

                          const auto order = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(1));
                          if (!order) {
                              OPENVINO_MLIR_DEBUG_PRINT("TransposePattern: rejected " << node->get_friendly_name() << " — non-constant order");
                              return false;
                          }

                          // An empty order means "reverse the dimensions" in OpenVINO, but linalg::TransposeOp
                          // needs an explicit permutation. A non-empty order is already validated to be a
                          // permutation of the input rank by the Transpose shape inference.
                          if (order->cast_vector<int64_t>().empty()) {
                              OPENVINO_MLIR_DEBUG_PRINT("TransposePattern: rejected " << node->get_friendly_name() << " — empty (implicit reverse) order");
                              return false;
                          }
                          return true;
                      }),
                  ConvertTranspose()) {}

// ConvertUnsqueeze requires Constant axes that are already normalized: non-negative, unique and
// sorted ascending.
UnsqueezePattern::UnsqueezePattern()
    : MarkPattern(
          wrap_type<v0::Unsqueeze>(
              {any_input(), any_input()},
              [](const Output<Node>& output) {
                  auto node = ov::as_type_ptr<v0::Unsqueeze>(output.get_node_shared_ptr());
                  if (!node || has_dynamic_rank(node)) {
                      OPENVINO_MLIR_DEBUG_PRINT("UnsqueezePattern: rejected " << output.get_node()->get_friendly_name() << " — dynamic rank");
                      return false;
                  }

                  const auto axes = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(1));
                  if (!axes) {
                      OPENVINO_MLIR_DEBUG_PRINT("UnsqueezePattern: rejected " << node->get_friendly_name() << " — non-constant axes");
                      return false;
                  }

                  // The converter reads the axes via Constant::get_coordinate_val(), which requires i64.
                  if (axes->get_element_type() != element::i64) {
                      OPENVINO_MLIR_DEBUG_PRINT("UnsqueezePattern: rejected " << node->get_friendly_name() << " — only i64 axes are supported, got "
                                                                              << axes->get_element_type());
                      return false;
                  }

                  const auto axes_values = axes->cast_vector<int64_t>();
                  if (axes_values.empty() || axes_values.front() < 0 || !std::is_sorted(axes_values.begin(), axes_values.end(), std::less_equal<int64_t>{})) {
                      OPENVINO_MLIR_DEBUG_PRINT("UnsqueezePattern: rejected " << node->get_friendly_name()
                                                                              << " — axes must be non-negative, unique and sorted ascending");
                      return false;
                  }
                  return true;
              }),
          ConvertUnsqueeze()) {}

BinaryEltwisePatternBase::BinaryEltwisePatternBase(NodeTypeInfo wrapped_type,
                                                   const GraphConverter::Convertor& convertor,
                                                   const std::set<element::Type>& element_types)
    : MarkPattern(std::make_shared<pass::pattern::op::WrapType>(
                      wrapped_type,
                      [element_types](const Output<Node>& output) {
                          if (!element_types.empty() && !element_types.count(output.get_element_type())) {
                              return false;
                          }
                          auto node = output.get_node_shared_ptr();
                          const auto inputs = node->inputs();
                          return std::all_of(inputs.begin(), inputs.end(), [&output](const Input<Node>& input) {
                              return statically_broadcastable(input.get_partial_shape(), output.get_partial_shape());
                          });
                      },
                      OutputVector{any_input(), any_input()}),
                  convertor) {}

template <typename OVOp, typename LinalgOp>
BinaryEltwisePattern<OVOp, LinalgOp>::BinaryEltwisePattern(const std::set<element::Type>& element_types)
    : BinaryEltwisePatternBase(OVOp::get_type_info_static(), ConvertBinaryEltwise<LinalgOp>(), element_types) {}

// Explicit template instantiations
// TODO: add signed/unsigned integers support
template class BinaryEltwisePattern<v1::Add, linalg::AddOp>;
template class BinaryEltwisePattern<v1::Subtract, linalg::SubOp>;
template class BinaryEltwisePattern<v1::Multiply, linalg::MulOp>;
template class BinaryEltwisePattern<v1::Divide, linalg::DivOp>;
template class BinaryEltwisePattern<v1::Power, linalg::PowFOp>;

template <typename OVOp, typename LinalgOp>
UnaryEltwisePattern<OVOp, LinalgOp>::UnaryEltwisePattern() : MarkPattern(wrap_type<OVOp>({any_input()}), ConvertUnaryEltwise<LinalgOp>()) {}

// Explicit template instantiations
template class UnaryEltwisePattern<v0::Abs, linalg::AbsOp>;
template class UnaryEltwisePattern<v0::Ceiling, linalg::CeilOp>;
template class UnaryEltwisePattern<v0::Exp, linalg::ExpOp>;
template class UnaryEltwisePattern<v0::Log, linalg::LogOp>;
template class UnaryEltwisePattern<v0::Negative, linalg::NegFOp>;
template class UnaryEltwisePattern<v0::Sqrt, linalg::SqrtOp>;
template class UnaryEltwisePattern<v0::Tanh, linalg::TanhOp>;

}  // namespace ov::intel_gpu::mlir
