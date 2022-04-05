// Copyright (C) 2018-2022 Intel Corporation
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
#include "ngraph/opsets/opset8.hpp"

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

bool propagateUpperBoundFromExistingDSR(const std::shared_ptr<ngraph::Function>& function) {
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
        {ngraph::opset5::Split::get_type_info_static(),         validateSplit},
        {ngraph::opset5::VariadicSplit::get_type_info_static(), validateSplit},
        {ngraph::opset6::Loop::get_type_info_static(),          validateLoop},
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
        {ngraph::opset3::Add::get_type_info_static(),                   dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Multiply::get_type_info_static(),              dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Subtract::get_type_info_static(),              dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::VariadicSplit::get_type_info_static(),         dynamicToStaticShapeVariadicSplit},
        {ngraph::opset3::Divide::get_type_info_static(),                dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Equal::get_type_info_static(),                 dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Greater::get_type_info_static(),               dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Power::get_type_info_static(),                 dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Maximum::get_type_info_static(),               dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Minimum::get_type_info_static(),               dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset3::Less::get_type_info_static(),                  dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset5::Select::get_type_info_static(),                dynamicToStaticShapeBinaryEltwise},
        {ngraph::opset5::NonMaxSuppression::get_type_info_static(),     dynamicToStaticNonMaxSuppression},
        {ngraph::opset3::NonZero::get_type_info_static(),               dynamicToStaticShapeNonZero},
        {ngraph::opset3::TopK::get_type_info_static(),                  dynamicToStaticShapeTopK},
        {ngraph::opset3::Transpose::get_type_info_static(),             dynamicToStaticShapeTranspose},
        {ngraph::opset3::Concat::get_type_info_static(),                dynamicToStaticShapeConcat},
        {ngraph::opset3::Convert::get_type_info_static(),               dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Clamp::get_type_info_static(),                 dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Floor::get_type_info_static(),                 dynamicToStaticUnaryElementwise},
        {ngraph::opset5::Ceiling::get_type_info_static(),               dynamicToStaticUnaryElementwise},
        {ngraph::opset5::Round::get_type_info_static(),                 dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Log::get_type_info_static(),                   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Relu::get_type_info_static(),                  dynamicToStaticUnaryElementwise},
        {ngraph::opset3::ScatterUpdate::get_type_info_static(),         dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Sigmoid::get_type_info_static(),               dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Softmax::get_type_info_static(),               dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Exp::get_type_info_static(),                   dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Sqrt::get_type_info_static(),                  dynamicToStaticUnaryElementwise},
        {ngraph::opset3::LogicalNot::get_type_info_static(),            dynamicToStaticUnaryElementwise},
        {ngraph::opset3::Abs::get_type_info_static(),                   dynamicToStaticUnaryElementwise},
        {ngraph::opset5::ScatterElementsUpdate::get_type_info_static(), dynamicToStaticUnaryElementwise},
        {ngraph::opset8::HSwish::get_type_info_static(),                dynamicToStaticUnaryElementwise},
        {ngraph::opset3::StridedSlice::get_type_info_static(),          dynamicToStaticShapeStridedSlice},
        {ngraph::opset3::Squeeze::get_type_info_static(),               dynamicToStaticShapeSqueeze},
        {ngraph::opset3::Gather::get_type_info_static(),                dynamicToStaticShapeGather},
        {ngraph::opset3::Unsqueeze::get_type_info_static(),             dynamicToStaticShapeUnsqueeze},
        {ngraph::opset3::ROIAlign::get_type_info_static(),              dynamicToStaticShapeROIAlign},
        {ngraph::opset3::Reshape::get_type_info_static(),               dynamicToStaticShapeReshape},
        {ngraph::opset3::Broadcast::get_type_info_static(),             dynamicToStaticShapeBroadcast},
        {ngraph::opset3::MatMul::get_type_info_static(),                dynamicToStaticShapeMatMul},
        {ngraph::opset5::Split::get_type_info_static(),                 dynamicToStaticShapeSplit},
        {ngraph::opset5::GatherND::get_type_info_static(),              dynamicToStaticShapeGatherND},
        {ngraph::opset6::GatherElements::get_type_info_static(),        dynamicToStaticShapeGatherElements},
        {ngraph::vpu::op::ExpGatherElements::get_type_info_static(),    dynamicToStaticShapeGatherElements},

        // reduction
        {ngraph::opset3::ReduceLogicalAnd::get_type_info_static(), dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceLogicalOr::get_type_info_static(),  dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMax::get_type_info_static(),        dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMean::get_type_info_static(),       dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceMin::get_type_info_static(),        dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceProd::get_type_info_static(),       dynamicToStaticShapeReduce},
        {ngraph::opset3::ReduceSum::get_type_info_static(),        dynamicToStaticShapeReduce},

        {ngraph::opset6::Loop::get_type_info_static(), dynamicToStaticShapeLoop},
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
    transformations.emplace(ngraph::opset3::Result::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>&){});
}

bool DynamicToStaticShape::run_on_model(const std::shared_ptr<ngraph::Function>& function) {
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
