// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/split.hpp"
#include "ngraph/node.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "low_precision/common/dequantization_op.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

SplitTransformation::SplitTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Split>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "SplitTransformation");
    this->register_matcher(m, callback);
}

bool SplitTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto split = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    const auto dequantization = NetworkHelper::getDequantization(split);

    OutputVector inputs = split->input_values();
    inputs[0] = dequantization.data;

    const auto newSplit = split->clone_with_new_inputs(inputs);
    newSplit->set_friendly_name(split->get_friendly_name());
    ngraph::copy_runtime_info(split, newSplit);

    const int64_t axis = as_type_ptr<opset1::Constant>(split->get_input_node_shared_ptr(1))->cast_vector<int64_t>()[0];
    const size_t normalizedAxis = normalize_axis(split->get_friendly_name(), axis, split->get_input_partial_shape(0).rank());
    const size_t outputSize = newSplit->get_output_size();

    const auto splitConstant = [&](const std::shared_ptr<Node> operation) {
        // if batch is absent in constant shape - add batch
        const auto normalizedConstant = NetworkHelper::normalizeDequantizationShape(operation);
        const auto constantShape = normalizedConstant->get_shape();

        OutputVector results(outputSize);
        if ((shape_size(constantShape) == 1ul) || (constantShape[normalizedAxis] == 1ul)) {
            std::for_each(results.begin(), results.end(), [&](Output<Node>& elem) { elem = normalizedConstant->clone_with_new_inputs({}); });
        } else {
            // prepare new inputs for constant folding
            OutputVector inputs = newSplit->input_values();
            inputs[0] = normalizedConstant;
            const auto foldSplit = newSplit->clone_with_new_inputs(inputs);

            // fold and fill results
            foldSplit->constant_fold(results, inputs);
        }

        for (auto& result : results) {
            result = NetworkHelper::toScalarIfPossible(result.get_node_shared_ptr());
        }

        return results;
    };

    // get splited dequantization constants
    OutputVector splitedSub = dequantization.subtract ? splitConstant(dequantization.subtract) : OutputVector{};
    OutputVector splitedMul = splitConstant(dequantization.multiply);

    NodeVector lastNodes;
    OutputVector replacement;
    for (size_t i = 0; i < outputSize; ++i) {
        Output<Node> parent = newSplit->output(i);

        if (dequantization.convert) {
            const auto convert = dequantization.convert->clone_with_new_inputs({ newSplit->output(i) });
            copy_runtime_info({ newSplit, convert }, convert);
            parent = convert;
        }

        if (dequantization.subtract) {
            const auto subtract = std::make_shared<DequantizationSubtract>(parent, splitedSub[i]);
            copy_runtime_info({ newSplit, subtract }, subtract);
            parent = subtract;
        }

        const auto multiply = std::make_shared<DequantizationMultiply>(parent, splitedMul[i]);
        copy_runtime_info({ newSplit, multiply }, multiply);

        lastNodes.push_back(multiply);
        replacement.push_back(multiply);
    }

    replace_node(split, replacement);
    updateOutputs(context, lastNodes, newSplit);
    return true;
}


void SplitTransformation::updateOutputs(
    TransformationContext& context,
    std::vector<std::shared_ptr<ngraph::Node>> lastNodes,
    std::shared_ptr<ngraph::Node> originalNode) const {
    // TODO: LPT: not implemented
    //const size_t outputSize = context.function->get_output_size();
    //if (outputSize == 1) {
    //    updateOutput(context, lastNodes[0], originalNode);
    //} else {
    //    const std::string originalName = originalNode->get_friendly_name();
    //    for (auto& lastNode : lastNodes) {
    //        for (size_t i = 0; i < outputSize; ++i) {
    //            std::shared_ptr<ngraph::Node> result = context.function->get_output_op(i);
    //            std::shared_ptr<ngraph::Node> outputNode = result->get_input_node_shared_ptr(0);
    //            if (outputNode.get() == lastNode.get()) {
    //                std::ostringstream oss;
    //                oss << i;
    //                originalNode->set_friendly_name(originalName + LayerTransformation::originalLayerPostfix);
    //                lastNode->set_friendly_name(originalName + "." + oss.str());
    //                break;
    //            }
    //        }
    //    }
    //}
}

bool SplitTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

bool SplitTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return (!NetworkHelper::getDequantization(layer).empty()) && LayerTransformation::canBeTransformed(context, layer);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
