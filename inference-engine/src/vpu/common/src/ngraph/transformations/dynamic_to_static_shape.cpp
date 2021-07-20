// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_binary_elementwise.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_broadcast.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_concat.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_gather.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_gather_elements.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_gather_nd.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_matmul.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_non_max_suppression.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_reduce.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_reshape.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_roialign.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_split.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_squeeze.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_strided_slice.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_topk.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_transpose.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_unary_elementwise.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_unsqueeze.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_variadic_split.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape_loop.hpp"

#include "vpu/ngraph/operations/exp_gather_elements.hpp"

#include "vpu/ngraph/utilities.hpp"
#include "vpu/utils/error.hpp"

#include "ngraph/opsets/opset3.hpp"
#include <ngraph/validation_util.hpp>
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset6.hpp"

namespace vpu {

namespace {

using namespace ngraph;

bool isDynamic(const Node& node) {
    const auto& outputs = node.outputs();
    return std::any_of(outputs.cbegin(), outputs.cend(), [](const Output<const Node>& output) {
        VPU_THROW_UNLESS(output.get_partial_shape().rank() != ngraph::Rank::dynamic(),
        "DynamicToStaticShape transformation: got dynamic rank for {} with type {} while only static is supported",
        output.get_node_shared_ptr()->get_friendly_name(), output.get_node_shared_ptr()->get_type_info());

        return output.get_partial_shape().is_dynamic();
    });
}

bool validateStaticShapes(const ngraph::Function& function) {
    for (const auto& node : function.get_ordered_ops()) {
        VPU_THROW_UNLESS(!isDynamic(*node),
            "DynamicToStaticShape transformation: after all the transformations there is still dynamism in the network."
            " First met node with dynamic output: {} (type: {})", node->get_friendly_name(), node->get_type_info());
    }
    return true;
}

bool propagateUpperBoundFromExistingDSR(std::shared_ptr<ngraph::Function>& function) {
    bool function_changed = false;
    for (const auto& op : function->get_ordered_ops()) {
        if (const auto dsr = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(op)) {
            dsr->setMode(ngraph::vpu::op::DynamicShapeResolverMode::INFER_UPPER_BOUND_SHAPE);
            dsr->validate_and_infer_types();
            function_changed = true;
        }
    }

    return function_changed;
}

using Validators = std::unordered_map<ngraph::DiscreteTypeInfo, std::function<void(const ngraph::Node&)>>;
const Validators& getValidators() {
    static const Validators validators = {
        {ngraph::opset5::Split::type_info,         validateSplit},
        {ngraph::opset5::VariadicSplit::type_info, validateSplit},
        {ngraph::opset6::Loop::type_info,          validateLoop},
    };
    return validators;
}

void validateDynamicFunction(const ngraph::Function& function) {
    const auto& validators = getValidators();
    for (const auto& node : function.get_ordered_ops()) {
        if (!validators.count(node->get_type_info())) {
            continue;
        }
        validators.at(node->get_type_info())(*node);
    }
}

const Transformations& getDefaultTransformations() {
    static const Transformations transformations = {
        {ngraph::opset3::Add::type_info,                   dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Multiply::type_info,              dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Subtract::type_info,              dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::VariadicSplit::type_info,         dynamicToStaticShapeVariadicSplit},
        {ngraph::opset3::Divide::type_info,                dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Equal::type_info,                 dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Greater::type_info,               dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Power::type_info,                 dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Maximum::type_info,               dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Minimum::type_info,               dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Less::type_info,                  dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset5::Select::type_info,                dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset5::NonMaxSuppression::type_info,     dynamicToStaticNonMaxSuppression},
        {ngraph::opset3::NonZero::type_info,               dynamicToStaticShapeNonZero},
        {ngraph::opset3::TopK::type_info,                  dynamicToStaticShapeTopK},
        {ngraph::opset3::Transpose::type_info,             dynamicToStaticShapeTranspose},
        {ngraph::opset3::Concat::type_info,                dynamicToStaticShapeConcat},
        {ngraph::opset3::Convert::type_info,               dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Clamp::type_info,                 dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Floor::type_info,                 dynamicToStaticUnaryElementwise},
        {ngraph::opset5::Ceiling::type_info,               dynamicToStaticUnaryElementwise},
        {ngraph::opset5::Round::type_info,                 dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Log::type_info,                   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Relu::type_info,                  dynamicToStaticUnaryElementwise},
        {ngraph::opset3::ScatterUpdate::type_info,         dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Sigmoid::type_info,               dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Softmax::type_info,               dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Exp::type_info,                   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Sqrt::type_info,                  dynamicToStaticUnaryElementwise},
        {ngraph::opset3::LogicalNot::type_info,            dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Abs::type_info,                   dynamicToStaticUnaryElementwise},
        {ngraph::opset5::ScatterElementsUpdate::type_info, dynamicToStaticUnaryElementwise},
        {ngraph::opset3::StridedSlice::type_info,          dynamicToStaticShapeStridedSlice},
        {ngraph::opset3::Squeeze::type_info,               dynamicToStaticShapeSqueeze},
        {ngraph::opset3::Gather::type_info,                dynamicToStaticShapeGather},
        {ngraph::opset3::Unsqueeze::type_info,             dynamicToStaticShapeUnsqueeze},
        {ngraph::opset3::ROIAlign::type_info,              dynamicToStaticShapeROIAlign},
        {ngraph::opset3::Reshape::type_info,               dynamicToStaticShapeReshape},
        {ngraph::opset3::Broadcast::type_info,             dynamicToStaticShapeBroadcast},
        {ngraph::opset3::MatMul::type_info,                dynamicToStaticShapeMatMul},
        {ngraph::opset5::Split::type_info,                 dynamicToStaticShapeSplit},
        {ngraph::opset5::GatherND::type_info,              dynamicToStaticShapeGatherND},
        {ngraph::opset6::GatherElements::type_info,        dynamicToStaticShapeGatherElements},
        {ngraph::vpu::op::ExpGatherElements::type_info,    dynamicToStaticShapeGatherElements},

        // reduction
        {ngraph::opset3::ReduceLogicalAnd::type_info, dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceLogicalOr::type_info,  dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMax::type_info,        dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMean::type_info,       dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMin::type_info,        dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceProd::type_info,       dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceSum::type_info,        dynamicToStaticShapeReduce},

        {ngraph::opset6::Loop::type_info, dynamicToStaticShapeLoop},
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

NGRAPH_RTTI_DEFINITION(DynamicToStaticShape, "DynamicToStaticShape", 0);

DynamicToStaticShape::DynamicToStaticShape(const Transformations& specificTransformations)
    : transformations(specificTransformations.empty() ? getDefaultTransformations() : specificTransformations) {
    transformations.emplace(ngraph::opset3::Result::type_info, [](const std::shared_ptr<ngraph::Node>&){});
}

bool DynamicToStaticShape::run_on_function(std::shared_ptr<ngraph::Function> function) {
    bool function_changed = false;

    // Ensure that existing DSRs in function propagate upper-bound shapes, not dynamism.
    // Basically this is possible in test cases, when the function is initially configured with DSR as inputs.
    function_changed |= propagateUpperBoundFromExistingDSR(function);

    // Operation-specific testing that needs to be performed in dynamic context before DSRs are introduced
    validateDynamicFunction(*function);

    // Make sure all values are invalidated, we need it to correctly evaluate upper-bound
    for (auto& node : function->get_ops()) {
        node->invalidate_values();
    }

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
        function_changed = true;
    }

    function->validate_nodes_and_infer_types();
    validateStaticShapes(*function);
    return function_changed;
}

}  // namespace vpu
