// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_inference/shape_infer_instances.hpp"
#include <openvino/op/util/unary_elementwise_arithmetic.hpp>
#include <openvino/op/util/binary_elementwise_arithmetic.hpp>
#include <openvino/op/util/binary_elementwise_comparison.hpp>
#include <openvino/op/util/binary_elementwise_logical.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <snippets/snippets_isa.hpp>

namespace ov {
namespace snippets {
using ShapeInferPtr = IShapeInferSnippetsFactory::ShapeInferPtr;

ShapeInferPtr IShapeInferSnippetsFactory::make(const ov::DiscreteTypeInfo& key, const std::shared_ptr<ov::Node>& op) {
    const auto& maker_iter = registry.find(key);
    if (maker_iter != registry.end())
        return maker_iter->second(op);
    return get_specific_op_shape_infer(key, op);
}

ShapeInferPtr IShapeInferSnippetsFactory::get_specific_op_shape_infer(const ov::DiscreteTypeInfo& key,
                                                                      const std::shared_ptr<ov::Node>& op) const {
    return {};
}

#define SHAPE_INFER_PREDEFINED(OP, InferType) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<InferType>();} }
#define SHAPE_INFER_OP_SPECIFIC(OP) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<OP::ShapeInfer>(n);} }
#define SHAPE_INFER_OP_SPECIFIC_EXTERNAL(OP, InferType) \
    { OP::get_type_info_static(), [](const std::shared_ptr<ov::Node>& n) { return std::make_shared<InferType>(n);} }

const IShapeInferSnippetsFactory::TRegistry IShapeInferSnippetsFactory::registry {
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
        //
        SHAPE_INFER_PREDEFINED(op::LoopBegin, SingleElementShapeInfer),
        SHAPE_INFER_PREDEFINED(op::Scalar, SingleElementShapeInfer),
        SHAPE_INFER_PREDEFINED(op::VectorBuffer, SingleElementShapeInfer),
        SHAPE_INFER_PREDEFINED(op::LoopEnd, EmptyShapeInfer),
#ifdef SNIPPETS_DEBUG_CAPS
        SHAPE_INFER_PREDEFINED(op::PerfCountBegin, EmptyShapeInfer),
        SHAPE_INFER_PREDEFINED(op::PerfCountEnd, EmptyShapeInfer),
#endif
        SHAPE_INFER_PREDEFINED(op::KernelStatic, EmptyShapeInfer),
        SHAPE_INFER_PREDEFINED(op::KernelDynamic, EmptyShapeInfer),
        SHAPE_INFER_PREDEFINED(op::Nop, EmptyShapeInfer),
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(op::Reshape, ReshapeShapeInfer),
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(opset1::Select, SelectShapeInfer),
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(op::Brgemm, BrgemmShapeInfer),
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(op::ReduceMax, ReduceShapeInfer),
        SHAPE_INFER_OP_SPECIFIC_EXTERNAL(op::ReduceSum, ReduceShapeInfer),
        // Note that Result has no output PortConnectors, so the shape must be empty
        SHAPE_INFER_PREDEFINED(ov::op::v0::Result, EmptyShapeInfer),
        //
        SHAPE_INFER_OP_SPECIFIC(op::LoadReshape),
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
    } else if (auto shape_infer = factory->make(op->get_type_info(), op)) {
        return shape_infer;
    } else if (ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(op)) {
        return std::make_shared<PassThroughShapeInfer>();
    } else if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseComparison>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseLogical>(op)) {
        return std::make_shared<NumpyBroadcastShapeInfer>();
    } else {
        OPENVINO_THROW("Operation type " + std::string(op->get_type_info().name) + " is not supported in Snippets shape inference pipeline");
    }
}

} // namespace snippets
} // namespace ov
