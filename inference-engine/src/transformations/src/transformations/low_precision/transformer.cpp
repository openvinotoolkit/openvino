// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/transformer.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ngraph_ops/type_relaxed.hpp"

#include "transformations/low_precision/convert.hpp"
#include "transformations/low_precision/convolution.hpp"
#include "transformations/low_precision/fake_quantize.hpp"

// uncomment to display precision info during low precision transformations
// #define DISPLAY_PECISION

namespace ngraph {
namespace pass {
namespace low_precision {

LowPrecisionTransformations::LowPrecisionTransformations(
    const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
    const std::map<std::string, LayerTransformationPtr>& transformations,
    const std::map<std::string, LayerTransformationPtr>& cleanupTransformations) :
    branchSpecificTransformations(branchSpecificTransformations),
    transformations(transformations),
    cleanupTransformations(cleanupTransformations) {}

void LowPrecisionTransformations::setUpdatePrecisions(const bool updatePrecisions) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
}

void LowPrecisionTransformations::setQuantizeOutputs(const bool quantizeOutputs) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizeOutputs(quantizeOutputs);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizeOutputs(quantizeOutputs);
    }
}

void LowPrecisionTransformations::setWeightsToConst(const bool weightsToConst) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setWeightsToConst(weightsToConst);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setWeightsToConst(weightsToConst);
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

LowPrecisionTransformations& LowPrecisionTransformations::remove(const std::string& layerName) {
    removeBranchSpecificTransformations(layerName);
    removeTransformations(layerName);
    removeCleanupTransformations(layerName);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeBranchSpecificTransformations(const std::string& layerName) {
    branchSpecificTransformations.erase(layerName);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeTransformations(const std::string& layerName) {
    transformations.erase(layerName);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeCleanupTransformations(const std::string& layerName) {
    cleanupTransformations.erase(layerName);
    return *this;
}

LayerTransformationPtr LowPrecisionTransformations::find(const std::string& transformationKey) const {
    auto it = branchSpecificTransformations.find(transformationKey);
    if (it != branchSpecificTransformations.end()) {
        return it->second;
    }

    it = transformations.find(transformationKey);
    if (it != transformations.end()) {
        return it->second;
    }

    it = cleanupTransformations.find(transformationKey);
    if (it != cleanupTransformations.end()) {
        return it->second;
    }

    return nullptr;
}

void LowPrecisionTransformations::setParamsManager(IParamsManager* paramsManager) noexcept {
    setParamsManager(paramsManager, branchSpecificTransformations);
    setParamsManager(paramsManager, transformations);
    setParamsManager(paramsManager, cleanupTransformations);
}

void LowPrecisionTransformations::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    setLayerTransformationsManager(layerTransformationsManager, branchSpecificTransformations);
    setLayerTransformationsManager(layerTransformationsManager, transformations);
    setLayerTransformationsManager(layerTransformationsManager, cleanupTransformations);
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setParamsManager(paramsManager);
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setLayerTransformationsManager(layerTransformationsManager);
    }
}

LowPrecisionTransformations LowPrecisionTransformer::getAllTransformations(const LayerTransformation::Params& params) {
    using namespace pass::low_precision;

    // TODO: refactor: duplication: declaration & registerMatcherIn
    return LowPrecisionTransformations().
        add<ConvolutionTransformation, opset1::Convolution>(params).
        add<FakeQuantizeTransformation, opset1::FakeQuantize>(params).

        // TODO: workaround: Convert I8 -> FP32 is not supported by CPU plugin
        addCleanup<ConvertTransformation, opset1::Convert>(params);
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
        if (!l_node) {
            std::cerr << "Error my matcher 1!!!\n";
            return false;
        }
        // std::cerr << "My matcher pass was triggered: " << l_node->get_friendly_name() << " with " << l_node->get_inputs().size() << " inputs\n";
        // TODO: replaces only operation with one output port
        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<BaseOp>>(*l_node, l_node->get_output_element_type(0));
        // auto replacement = std::make_shared<BaseOp>(*l_node);
        copy_runtime_info(l_node, replacement);
        replace_node(l_node, replacement);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "TypeRelaxedReplacer");
    transformation->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

TypeRelaxedReplacer::TypeRelaxedReplacer() {
    // List all operations that support polymorphic inputs/outputs
    make_matcher_type_relaxed<opset1::AvgPool>(this);
    make_matcher_type_relaxed<opset1::Concat>(this);
    make_matcher_type_relaxed<opset1::Convolution>(this);
    make_matcher_type_relaxed<opset1::FakeQuantize>(this);
    make_matcher_type_relaxed<opset1::GroupConvolution>(this);
    make_matcher_type_relaxed<opset1::Relu>(this);
    make_matcher_type_relaxed<opset1::MaxPool>(this);
    make_matcher_type_relaxed<opset1::Add>(this);
    make_matcher_type_relaxed<opset1::Subtract>(this);
    make_matcher_type_relaxed<ngraph::op::Subtract>(this);
    make_matcher_type_relaxed<opset1::NormalizeL2>(this);
    make_matcher_type_relaxed<opset1::Multiply>(this);
}

LowPrecisionTransformer::LowPrecisionTransformer(const LowPrecisionTransformations& transformations)
    : transformations(transformations) {}

void LowPrecisionTransformer::transform(std::shared_ptr<Function> network) {
    transformations.setParamsManager(this);
    transformations.setLayerTransformationsManager(this);

    TransformationContext context(network);

    // Extend necessary operations with polymorphic semantics
    {
        TypeRelaxedReplacer pass;
        pass.run_on_function(network);
    }

    {
        // Branch specific transformations
        GraphRewrite pass;
        registerAllMatchers(transformations.branchSpecificTransformations, pass, context);
        pass.run_on_function(network);
    }

    {
        // Step #1: FakeQuantize layer transformation execution
        LayerTransformationPtr fqTransformation = transformations.find<opset1::FakeQuantize>();
        if (fqTransformation == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "FakeQuantize transformation was not found";
        }
        GraphRewrite pass;
        fqTransformation->registerMatcherIn(pass, context);
        pass.run_on_function(network);
    }

    {
        // Step #2: layer transformations execution
        GraphRewrite pass;
        registerAllMatchers(transformations.transformations, pass, context);
        pass.run_on_function(network);
    }

    {
        // Step #3: cleanup transformations execution
        GraphRewrite pass;
        registerAllMatchers(transformations.cleanupTransformations, pass, context);
        pass.run_on_function(network);
    }
}

std::vector<element::Type> LowPrecisionTransformer::getPrecisionsOnActivations(const Node& op) const noexcept {
    const LayerTransformationPtr transformation = transformations.find(LowPrecisionTransformations::getType(op));
    if (transformation == nullptr) {
        return std::vector<element::Type>();
    }
    return transformation->getPrecisionsOnActivations();
}

bool LowPrecisionTransformer::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(*layer);

    const LayerTransformationPtr transformation = transformations.find(operantionType);
    if (transformation == nullptr) {
        return false;
    }
    return transformation->isQuantized(layer);
}

bool LowPrecisionTransformer::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    const std::string operantionType = LowPrecisionTransformations::getType(*layer);
    const LayerTransformationPtr transformation = transformations.find(operantionType);
    if (transformation == nullptr) {
        return false;
    }
    return transformation->isPrecisionPreserved(layer);
}

void LowPrecisionTransformer::registerAllMatchers(
    std::map<std::string, LayerTransformationPtr> transformations,
    GraphRewrite& pass,
    TransformationContext& context) {
    for (auto it : transformations) {
        it.second->registerMatcherIn(pass, context);
    }
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
