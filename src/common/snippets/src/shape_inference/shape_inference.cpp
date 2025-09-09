// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets/shape_inference/shape_inference.hpp"

#include <memory>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/util/binary_elementwise_arithmetic.hpp>
#include <openvino/op/util/binary_elementwise_comparison.hpp>
#include <openvino/op/util/binary_elementwise_logical.hpp>
#include <openvino/op/util/unary_elementwise_arithmetic.hpp>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/op/fill.hpp"
#include "snippets/op/horizon_max.hpp"
#include "snippets/op/horizon_sum.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/nop.hpp"
#include "snippets/op/online_softmax.hpp"
#include "snippets/op/online_softmax_update_max.hpp"
#include "snippets/op/online_softmax_update_sum.hpp"
#include "snippets/op/perf_count.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/reg_spill.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"
#include "snippets/op/vector_buffer.hpp"
#include "snippets/shape_inference/shape_infer_instances.hpp"

namespace ov::snippets {
using ShapeInferPtr = IShapeInferSnippetsFactory::ShapeInferPtr;

ShapeInferPtr IShapeInferSnippetsFactory::make(const ov::DiscreteTypeInfo& key, const std::shared_ptr<ov::Node>& op) {
    const auto& maker_iter = registry.find(key);
    if (maker_iter != registry.end()) {
        return maker_iter->second(op);
    }
    return get_specific_op_shape_infer(key, op);
}

ShapeInferPtr IShapeInferSnippetsFactory::get_specific_op_shape_infer(
    [[maybe_unused]] const ov::DiscreteTypeInfo& key,
    [[maybe_unused]] const std::shared_ptr<ov::Node>& op) const {
    return {};
}

#define SHAPE_INFER_PREDEFINED(OP, InferType)                                              \
    {OP::get_type_info_static(), []([[maybe_unused]] const std::shared_ptr<ov::Node>& n) { \
         return std::make_shared<InferType>();                                             \
     }}
#define SHAPE_INFER_OP_SPECIFIC(OP)                                       \
    {OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { \
         return std::make_shared<OP::ShapeInfer>(n);                      \
     }}
#define SHAPE_INFER_OP_SPECIFIC_EXTERNAL(OP, InferType)                   \
    {OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { \
         return std::make_shared<InferType>(n);                           \
     }}

const IShapeInferSnippetsFactory::TRegistry IShapeInferSnippetsFactory::registry{
    SHAPE_INFER_PREDEFINED(op::ConvertTruncation, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(op::ConvertSaturation, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(op::Load, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(op::Store, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(op::Fill, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(ov::op::v0::Parameter, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(ov::op::v1::LogicalNot, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(ov::op::v0::PRelu, PassThroughShapeInfer),
    SHAPE_INFER_PREDEFINED(op::HorizonMax, HorizonOpShapeInfer),
    SHAPE_INFER_PREDEFINED(op::HorizonSum, HorizonOpShapeInfer),
    SHAPE_INFER_PREDEFINED(op::OnlineSoftmax, OnlineSoftmaxShapeInfer),
    SHAPE_INFER_PREDEFINED(op::OnlineSoftmaxUpdateMax, OnlineSoftmaxUpdateMaxShapeInfer),
    SHAPE_INFER_PREDEFINED(op::OnlineSoftmaxUpdateSum, OnlineSoftmaxUpdateSumShapeInfer),
    //
    SHAPE_INFER_PREDEFINED(op::Scalar, SingleElementShapeInfer),
    SHAPE_INFER_PREDEFINED(op::VectorBuffer, SingleElementShapeInfer),
    SHAPE_INFER_PREDEFINED(op::LoopBegin, SingleElementShapeInfer),
    SHAPE_INFER_PREDEFINED(op::LoopEnd, EmptyShapeInfer),
    SHAPE_INFER_OP_SPECIFIC(op::RegSpillBegin),
    SHAPE_INFER_PREDEFINED(op::RegSpillEnd, EmptyShapeInfer),
#ifdef SNIPPETS_DEBUG_CAPS
    SHAPE_INFER_PREDEFINED(op::PerfCountBegin, EmptyShapeInfer),
    SHAPE_INFER_PREDEFINED(op::PerfCountEnd, EmptyShapeInfer),
#endif
    SHAPE_INFER_PREDEFINED(op::KernelStatic, EmptyShapeInfer),
    SHAPE_INFER_PREDEFINED(op::KernelDynamic, EmptyShapeInfer),
    SHAPE_INFER_PREDEFINED(op::Nop, EmptyShapeInfer),
    SHAPE_INFER_OP_SPECIFIC_EXTERNAL(opset1::Select, SelectShapeInfer),
    SHAPE_INFER_OP_SPECIFIC_EXTERNAL(op::Brgemm, BrgemmShapeInfer),
    SHAPE_INFER_OP_SPECIFIC_EXTERNAL(op::ReduceMax, ReduceShapeInfer),
    SHAPE_INFER_OP_SPECIFIC_EXTERNAL(op::ReduceSum, ReduceShapeInfer),
    // Note that Result has no output PortConnectors, so the shape must be empty
    SHAPE_INFER_PREDEFINED(ov::op::v0::Result, EmptyShapeInfer),
    //
    SHAPE_INFER_OP_SPECIFIC(op::LoadReorder),
    SHAPE_INFER_OP_SPECIFIC(op::Reshape),
    SHAPE_INFER_OP_SPECIFIC(op::Reorder),
    SHAPE_INFER_OP_SPECIFIC(op::RankNormalization),
    SHAPE_INFER_OP_SPECIFIC(op::BroadcastLoad),
    SHAPE_INFER_OP_SPECIFIC(op::BroadcastMove),
    SHAPE_INFER_OP_SPECIFIC(op::Buffer),
};
#undef SHAPE_INFER_OP_SPECIFIC_EXTERNAL
#undef SHAPE_INFER_OP_SPECIFIC
#undef SHAPE_INFER_PREDEFINED

std::shared_ptr<IShapeInferSnippets> make_shape_inference(const std::shared_ptr<ov::Node>& op,
                                                          const std::shared_ptr<IShapeInferSnippetsFactory>& factory) {
    if (!factory) {
        return nullptr;
    }
    if (auto shape_infer = factory->make(op->get_type_info(), op)) {
        return shape_infer;
    }
    if (ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(op)) {
        return std::make_shared<PassThroughShapeInfer>();
    }
    if (ov::is_type_any_of<ov::op::util::BinaryElementwiseArithmetic,
                           ov::op::util::BinaryElementwiseComparison,
                           ov::op::util::BinaryElementwiseLogical>(op)) {
        return std::make_shared<NumpyBroadcastShapeInfer>();
    }
    OPENVINO_THROW("Operation type " + std::string(op->get_type_info().name) +
                   " is not supported in Snippets shape inference pipeline");
}

}  // namespace ov::snippets
