// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_binary_elementwise.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_broadcast.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_concat.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_gather.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_matmul.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_non_max_suppression.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_reduce.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_reshape.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_roialign.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_squeeze.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_strided_slice.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_topk.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_transpose.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_unary_elementwise.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_unsqueeze.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_variadic_split.hpp"

#include "vpu/utils/error.hpp"

#include "ngraph/opsets/opset3.hpp"

namespace vpu {

void printTo(std::ostream& stream, const ngraph::NodeTypeInfo& object) {
    stream << object.name << " ver. " << object.version;
}

namespace {

using namespace ngraph;

bool isDynamic(const Node& node) {
    const auto& outputs = node.outputs();
    return std::any_of(outputs.cbegin(), outputs.cend(), [](const Output<const Node>& output) {
        VPU_THROW_UNLESS(output.get_partial_shape().rank() != ngraph::Rank::dynamic(),
        "DynamicToStaticShape transformation: got dynamic rank for {} with type {} while only static is supported",
        output.get_node_shared_ptr()->get_friendly_name(), output.get_node_shared_ptr()->get_type_name());

        return output.get_partial_shape().is_dynamic();
    });
}

bool validateStaticShapes(const ngraph::Function& function) {
    for (const auto& node : function.get_ordered_ops()) {
        VPU_THROW_UNLESS(!isDynamic(*node),
            "DynamicToStaticShape transformation: after all the transformations there is still dynamism in the network."
            " First met node with dynamic output: {} (type: {})", node->get_friendly_name(), node->get_type_name());
    }
    return true;
}

const Transformations& getDefaultTransformations() {
    static const Transformations transformations = {
        {ngraph::opset3::Add::type_info, dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Multiply::type_info, dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Subtract::type_info, dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::VariadicSplit::type_info, dynamicToStaticShapeVariadicSplit},
        {ngraph::opset3::Divide::type_info, dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Equal::type_info, dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Greater::type_info, dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Power::type_info, dynamicToStaticShapeBinaryEltwise},
        {ngraph::op::dynamic::NonMaxSuppression::type_info, dynamicToStaticNonMaxSuppression},
        {ngraph::opset3::NonZero::type_info,   dynamicToStaticShapeNonZero},
        {ngraph::opset3::TopK::type_info, dynamicToStaticShapeTopK},
        {ngraph::opset3::Transpose::type_info, dynamicToStaticShapeTranspose},
        {ngraph::opset3::Concat::type_info,    dynamicToStaticShapeConcat},
        {ngraph::opset3::Convert::type_info,   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Clamp::type_info,     dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Floor::type_info,     dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Log::type_info,       dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Relu::type_info,      dynamicToStaticUnaryElementwise},
        {ngraph::opset3::ScatterUpdate::type_info, dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Sigmoid::type_info,   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Softmax::type_info,   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Exp::type_info,   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Sqrt::type_info,      dynamicToStaticUnaryElementwise},
        {ngraph::opset3::StridedSlice::type_info,   dynamicToStaticShapeStridedSlice},
        {ngraph::opset3::Squeeze::type_info,   dynamicToStaticShapeSqueeze},
        {ngraph::opset3::Gather::type_info,    dynamicToStaticShapeGather},
        {ngraph::opset3::Unsqueeze::type_info, dynamicToStaticShapeUnsqueeze},
        {ngraph::opset3::ROIAlign::type_info,  dynamicToStaticShapeROIAlign},
        {ngraph::opset3::Reshape::type_info,   dynamicToStaticShapeReshape},
        {ngraph::opset3::Broadcast::type_info, dynamicToStaticShapeBroadcast},
        {ngraph::opset3::MatMul::type_info, dynamicToStaticShapeMatMul},

        // reduction
        {ngraph::opset3::ReduceLogicalAnd::type_info, dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceLogicalOr::type_info, dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMax::type_info, dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMean::type_info, dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMin::type_info, dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceProd::type_info, dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceSum::type_info, dynamicToStaticShapeReduce},
    };
    return transformations;
}

std::set<NodeTypeInfo> getSupportedTypes(const Transformations& transformations) {
    auto supportedTypes = std::set<NodeTypeInfo>{};
    for (const auto& transformation : transformations) {
        supportedTypes.insert(transformation.first);
    }
    return supportedTypes;
}

}  // namespace

DynamicToStaticShape::DynamicToStaticShape(const Transformations& specificTransformations)
    : transformations(specificTransformations.empty() ? getDefaultTransformations() : specificTransformations) {
    transformations.emplace(ngraph::opset3::Result::type_info, [](const std::shared_ptr<ngraph::Node>&){});
}

void DynamicToStaticShape::transform(std::shared_ptr<ngraph::Function> function) const {
    for (const auto& operation : function->get_ordered_ops()) {
        if (!isDynamic(*operation)) {
            continue;
        }

        const auto type = operation->get_type_info();
        const auto transformation = transformations.find(type);
        VPU_THROW_UNLESS(transformation != transformations.cend(),
            "DynamicToStaticShape transformation encountered dynamic node {} of type {}, but only {} types are supported for dynamic nodes",
            operation->get_friendly_name(), type, getSupportedTypes(transformations));
        transformation->second(operation);
    }

    function->validate_nodes_and_infer_types();
    validateStaticShapes(*function);
}

}  // namespace vpu
