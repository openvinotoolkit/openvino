// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transformer.hpp"
#include "low_precision/network_helper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/opsets/opset6.hpp"

#include "lpt_itt.h"

// branch specific transformations
#include "low_precision/concat.hpp"
#include "low_precision/concat_multi_channels.hpp"

// decomposition transformations
#include "low_precision/fake_quantize_decomposition.hpp"

// general transformations
#include "low_precision/add.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/multiply.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/reduce_max.hpp"
#include "low_precision/reduce_mean.hpp"
#include "low_precision/reduce_min.hpp"
#include "low_precision/reduce_sum.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/shuffle_channels.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/subtract.hpp"
#include "low_precision/split.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"
#include "low_precision/split.hpp"

// cleanup transformations
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/fuse_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

LowPrecisionTransformations::LowPrecisionTransformations(
    const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
    const std::map<std::string, LayerTransformationPtr>& decompositionTransformations,
    const std::map<std::string, LayerTransformationPtr>& transformations,
    const std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>>& cleanupTransformations,
    const std::vector<StandaloneCleanup>& standaloneCleanupTransformations) :
    branchSpecificTransformations(branchSpecificTransformations),
    decompositionTransformations(decompositionTransformations),
    transformations(transformations),
    cleanupTransformations(cleanupTransformations),
    standaloneCleanupTransformations(standaloneCleanupTransformations) {}

void LowPrecisionTransformations::setUpdatePrecisions(const bool updatePrecisions) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnActivations(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnWeights(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
}

std::vector<LayerTransformationPtr> LowPrecisionTransformations::find(const std::string& transformationKey) const {
    auto it = branchSpecificTransformations.find(transformationKey);
    std::vector<LayerTransformationPtr> res;
    if (it != branchSpecificTransformations.end()) {
        res.emplace_back(it->second);
    }

    it = transformations.find(transformationKey);
    if (it != transformations.end()) {
        res.emplace_back(it->second);
    }

    const auto it1 = cleanupTransformations.find(transformationKey);
    if (it1 != cleanupTransformations.end()) {
        for (const auto& transformation : it1->second) {
            res.emplace_back(transformation.second);
        }
    }

    for (const auto& transformation : standaloneCleanupTransformations) {
        if (transformation.typeName == transformationKey) {
            res.emplace_back(transformation.transformation);
        }
    }

    return res;
}

void LowPrecisionTransformations::setParamsManager(IParamsManager* paramsManager) noexcept {
    setParamsManager(paramsManager, branchSpecificTransformations);
    setParamsManager(paramsManager, decompositionTransformations);
    setParamsManager(paramsManager, transformations);
    setParamsManager(paramsManager, cleanupTransformations);
    setParamsManager(paramsManager, standaloneCleanupTransformations);
}

void LowPrecisionTransformations::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    setLayerTransformationsManager(layerTransformationsManager, branchSpecificTransformations);
    setLayerTransformationsManager(layerTransformationsManager, decompositionTransformations);
    setLayerTransformationsManager(layerTransformationsManager, transformations);
    setLayerTransformationsManager(layerTransformationsManager, cleanupTransformations);
    setLayerTransformationsManager(layerTransformationsManager, standaloneCleanupTransformations);
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setParamsManager(paramsManager);
    }
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>>& transformations) noexcept {
    for (auto it : transformations) {
        for (auto transform : it.second) {
            transform.second->setParamsManager(paramsManager);
        }
    }
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::vector<StandaloneCleanup>& transformations) noexcept {
    for (auto it : transformations) {
        it.transformation->setParamsManager(paramsManager);
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setLayerTransformationsManager(layerTransformationsManager);
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::map < std::string, std::vector < std::pair<std::string,  LayerTransformationPtr >> > & transformations) noexcept {
    for (auto it : transformations) {
        for (auto transform : it.second) {
            transform.second->setLayerTransformationsManager(layerTransformationsManager);
        }
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::vector<StandaloneCleanup>& transformations) noexcept {
    for (auto it : transformations) {
        it.transformation->setLayerTransformationsManager(layerTransformationsManager);
    }
}

LowPrecisionTransformations LowPrecisionTransformer::getAllTransformations(const LayerTransformation::Params& params) {
    using namespace pass::low_precision;

    auto transformer = LowPrecisionTransformations().
        addBranchSpecific<pass::low_precision::ConcatMultiChannelsTransformation, opset1::Concat>(params).

        addDecomposition<pass::low_precision::FakeQuantizeDecompositionTransformation, opset1::FakeQuantize>(params).

        add<AddTransformation, opset1::Add>(params).
        add<AvgPoolTransformation, opset1::AvgPool>(params).
        add<ClampTransformation, opset1::Clamp>(params).
        add<ConvolutionTransformation, opset1::Convolution>(params).
        add<ConvolutionBackpropDataTransformation, opset1::ConvolutionBackpropData>(params).
        add<DepthToSpaceTransformation, opset1::DepthToSpace>(params).
        add<FakeQuantizeTransformation, opset1::FakeQuantize>(params).
        add<GroupConvolutionTransformation, opset1::GroupConvolution>(params).
        add<InterpolateTransformation, opset1::Interpolate>(params).
        add<InterpolateTransformation, opset4::Interpolate>(params).
        add<MatMulTransformation, opset1::MatMul>(params).
        add<MaxPoolTransformation, opset1::MaxPool>(params).
        add<MultiplyTransformation, opset1::Multiply>(params).
        add<MVNTransformation, op::MVN>(params).
        add<MVNTransformation, opset6::MVN>(params).
        add<NormalizeL2Transformation, opset1::NormalizeL2>(params).
        add<PReluTransformation, opset1::PRelu>(params).
        add<ReduceMaxTransformation, opset1::ReduceMax>(params).
        add<ReduceMeanTransformation, opset1::ReduceMean>(params).
        add<ReduceMinTransformation, opset1::ReduceMin>(params).
        add<ReduceSumTransformation, opset1::ReduceSum>(params).
        add<ReluTransformation, opset1::Relu>(params).
        add<ReshapeTransformation, opset1::Reshape>(params).
        add<ShuffleChannelsTransformation, opset1::ShuffleChannels>(params).
        add<SqueezeTransformation, opset1::Squeeze>(params).
        add<SplitTransformation, opset1::Split>(params).
        add<StridedSliceTransformation, opset1::StridedSlice>(params).
        add<TransposeTransformation, opset1::Transpose>(params).
        add<UnsqueezeTransformation, opset1::Unsqueeze>(params).
        add<VariadicSplitTransformation, opset1::VariadicSplit>(params).

        addCleanup<FoldConvertTransformation, opset1::Subtract>(params).
        addCleanup<FuseConvertTransformation, opset1::Multiply>(params).

        addStandaloneCleanup<FuseSubtractToFakeQuantizeTransformation, opset1::Subtract>(params).
        addStandaloneCleanup<FuseMultiplyToFakeQuantizeTransformation, opset1::Multiply>(params).
        addStandaloneCleanup<MultiplyToGroupConvolutionTransformation, opset1::Multiply>(params).
        addStandaloneCleanup<SubtractMultiplyToMultiplyAddTransformation, opset1::Multiply>(params);

    return transformer;
}

bool LowPrecisionTransformer::isFunctionQuantized(const std::shared_ptr<const Function>& function) {
    std::set<std::shared_ptr<Node>> handledNodes;
    std::deque<std::shared_ptr<Node>> nodes;
    for (auto result : function->get_results()) {
        nodes.push_front(result);
    }

    while (!nodes.empty()) {
        auto node = nodes.front();
        nodes.pop_front();

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            auto parent = node->get_input_node_shared_ptr(i);
            if (handledNodes.find(parent) != handledNodes.end()) {
                continue;
            }

            const std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
            if ((fakeQuantize != nullptr) &&
                QuantizationDetails::outputLayoutIsSupported(fakeQuantize) &&
                QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels())) {
                return true;
            }

            nodes.push_front(parent);
            handledNodes.insert(parent);
        }
    }
    return false;
}

LowPrecisionTransformer::LowPrecisionTransformer(): transformations(LowPrecisionTransformer::getAllTransformations()) {}

template <typename BaseOp>
void make_matcher_type_relaxed(ngraph::pass::GraphRewrite* transformation) {
    using namespace ngraph;

    auto is_op_type = [](std::shared_ptr<Node> n) {
        return !!as_type_ptr<BaseOp>(n);
    };

    auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, is_op_type);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) {
        auto l_node = std::dynamic_pointer_cast<BaseOp>(m.get_match_root());
        if (std::dynamic_pointer_cast<op::TypeRelaxedBase>(l_node)) {
            return false;
        }
        if (!l_node) {
            THROW_IE_LPT_EXCEPTION(*l_node) << "unexpected operation type";
        }

        std::vector<element::Type> inputPrecisions;
        for (auto& inputs : l_node->inputs()) {
            inputPrecisions.push_back(inputs.get_element_type());
        }

        std::vector<element::Type> outputPrecisions;
        for (auto& output : l_node->outputs()) {
            outputPrecisions.push_back(output.get_element_type());
        }

        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<BaseOp>>(*l_node, inputPrecisions, outputPrecisions);

        copy_runtime_info(l_node, replacement);
        replace_node(l_node, replacement);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "TypeRelaxedReplacer");
    NGRAPH_SUPPRESS_DEPRECATED_START
    transformation->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}

TypeRelaxedReplacer::TypeRelaxedReplacer() {
    make_matcher_type_relaxed<opset1::Add>(this);
    make_matcher_type_relaxed<opset1::AvgPool>(this);
    make_matcher_type_relaxed<opset1::Clamp>(this);
    make_matcher_type_relaxed<opset1::Concat>(this);
    make_matcher_type_relaxed<opset1::Convolution>(this);
    make_matcher_type_relaxed<opset1::ConvolutionBackpropData>(this);
    make_matcher_type_relaxed<opset1::DepthToSpace>(this);
    make_matcher_type_relaxed<opset1::FakeQuantize>(this);
    make_matcher_type_relaxed<opset1::GroupConvolution>(this);
    make_matcher_type_relaxed<opset1::PRelu>(this);
    make_matcher_type_relaxed<opset1::ReduceMean>(this);
    make_matcher_type_relaxed<opset1::ReduceSum>(this);
    make_matcher_type_relaxed<opset1::Subtract>(this);
    make_matcher_type_relaxed<opset1::Interpolate>(this);
    make_matcher_type_relaxed<opset1::Multiply>(this);
    make_matcher_type_relaxed<op::MVN>(this);
    make_matcher_type_relaxed<opset6::MVN>(this);
    make_matcher_type_relaxed<opset1::NormalizeL2>(this);
    make_matcher_type_relaxed<opset4::Interpolate>(this);
}

LowPrecisionTransformer::LowPrecisionTransformer(const LowPrecisionTransformations& transformations)
    : transformations(transformations) {}

void LowPrecisionTransformer::transform(std::shared_ptr<Function> network) {
    if (!isFunctionQuantized(network)) {
        return;
    }

    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::LPT_LT, "LowPrecisionTransformer", "transform");

    ngraph::pass::ConstantFolding constantFolding;
    constantFolding.run_on_function(network);

    transformations.setParamsManager(this);
    transformations.setLayerTransformationsManager(this);

    TransformationContext context(network);

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "TypeRelaxedReplacer");

    // Extend necessary operations with polymorphic semantics
    {
        TypeRelaxedReplacer pass;
        pass.run_on_function(network);
    }

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "BranchSpecificTransformations");

    {
        // Branch specific transformations
        GraphRewrite pass;
        registerAllMatchers(transformations.branchSpecificTransformations, pass, context);
        pass.run_on_function(network);
    }

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "FakeQuantizeDecomposition");

    {
        // Step #1: FakeQuantize decomposition transformation execution
        GraphRewrite pass;
        registerAllMatchers(transformations.decompositionTransformations, pass, context);
        pass.run_on_function(network);
    }

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "LayerTransformations");

    {
        // Step #2: layer transformations execution
        GraphRewrite pass;
        registerAllMatchers(transformations.transformations, pass, context);
        pass.run_on_function(network);
    }

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "CleanupTransformations");

    {
        // Step #3: cleanup transformations execution
        GraphRewrite pass;
        registerAllMatchers(transformations.cleanupTransformations, pass, context);
        pass.run_on_function(network);
    }

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "StandaloneCleanupTransformations");

    {
        // Step #4: standalone cleanup transformations execution

        for (auto it : transformations.standaloneCleanupTransformations) {
            GraphRewrite pass;
            it.transformation->registerMatcherIn(pass, context);
            pass.run_on_function(network);
        }
    }

    network->validate_nodes_and_infer_types();
}

std::vector<element::Type> LowPrecisionTransformer::getPrecisionsOnActivations(const Node& op) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(op);
    const std::vector<LayerTransformationPtr> transformation = transformations.find(operantionType);
    if (transformation.empty()) {
        return std::vector<element::Type>();
    }
    std::vector<element::Type> precisions = transformation[0]->getPrecisionsOnActivations();

    for (const auto& transform : transformation) {
        precisions = NetworkHelper::precisionIntersection(precisions, transform->getPrecisionsOnActivations());
    }
    return precisions;
}

bool LowPrecisionTransformer::isQuantized(const std::shared_ptr<Node>& layer) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(*layer);
    const std::vector<LayerTransformationPtr> transformation = transformations.find(operantionType);
    if (transformation.empty()) {
        return false;
    }

    for (const auto& transform : transformation) {
        if (!transform->isQuantized(layer)) {
            return false;
        }
    }
    return true;
}

bool LowPrecisionTransformer::isPrecisionPreserved(const std::shared_ptr<Node>& layer) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(*layer);
    const std::vector<LayerTransformationPtr> transformation = transformations.find(operantionType);
    if (transformation.empty()) {
        return false;
    }

    for (const auto& transform : transformation) {
        if (!transform->isPrecisionPreserved(layer)) {
            return false;
        }
    }
    return true;
}

void LowPrecisionTransformer::registerAllMatchers(
    std::map<std::string, LayerTransformationPtr> transformations,
    GraphRewrite& pass,
    TransformationContext& context) {
    for (auto it : transformations) {
        it.second->registerMatcherIn(pass, context);
    }
}

void LowPrecisionTransformer::registerAllMatchers(
    std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>> transformations,
    GraphRewrite& pass,
    TransformationContext& context) {
    for (auto it : transformations) {
        for (auto transform : it.second) {
            transform.second->registerMatcherIn(pass, context);
        }
    }
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
