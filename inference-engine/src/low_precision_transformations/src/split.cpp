// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/split.hpp"
#include "ngraph/node.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/common/dequantization_op.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {
SplitTransformation::SplitTransformation(const Params& params) : LayerTransformation(params) {}

void SplitTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::Split>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
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

    int64_t axis = as_type_ptr<opset1::Constant>(split->get_input_node_shared_ptr(1))->cast_vector<int64_t>()[0];
    size_t normalizeAxis = normalize_axis(split->get_friendly_name(), axis, split->get_input_partial_shape(0).rank());

    NodeVector lastNodes;
    OutputVector replacement;
    for (size_t i = 0; i < newSplit->get_output_size(); ++i) {
        Output<Node> previous = newSplit->output(i);

        const auto splitConstant = [&](const std::shared_ptr<opset1::Constant> constant, const std::shared_ptr<Node> eltwise) {
            if (constant->get_shape().empty()) {
                return constant;
            } else {
                // if batch is absent in constant shape - add batch
                const auto normalizedConstant = NetworkHelper::normalizeDequantizationShape(eltwise);
                if (normalizedConstant->get_shape()[normalizeAxis] == 1) {
                    return normalizedConstant;
                } else {
                    return NetworkHelper::foldDequantizationConstant(newSplit, normalizedConstant, i);
                }
            }
        };

        if (dequantization.convert) {
            const auto convert = dequantization.convert->clone_with_new_inputs({ newSplit->output(i) });
            copy_runtime_info({ newSplit, convert }, convert);
            previous = convert;
        }

        if (dequantization.subtract) {
            const auto subConst = splitConstant(dequantization.subtractConstant, dequantization.subtract);
            const auto subtract = std::make_shared<DequantizationSubtract>(previous, subConst);

            copy_runtime_info({ newSplit, subtract }, subtract);
            previous = subtract;
        }

        if (dequantization.multiply) {
            const auto mulConst = splitConstant(dequantization.multiplyConstant, dequantization.multiply);
            const auto multiply = std::make_shared<DequantizationMultiply>(previous, mulConst);

            copy_runtime_info({ newSplit, multiply }, multiply);
            previous = multiply;
        }

        lastNodes.push_back(previous.get_node_shared_ptr());
        replacement.push_back(previous);
    }

    replace_node(split, replacement);
    updateOutputs(context, lastNodes, newSplit);
    return true;
}


void SplitTransformation::updateOutputs(
    TransformationContext& context,
    std::vector<std::shared_ptr<ngraph::Node>> lastNodes,
    std::shared_ptr<ngraph::Node> originalNode) const {
    const size_t outputSize = context.function->get_output_size();
    if (outputSize == 1) {
        updateOutput(context, lastNodes[0], originalNode);
    } else {
        const std::string originalName = originalNode->get_friendly_name();
        for (auto& lastNode : lastNodes) {
            for (size_t i = 0; i < outputSize; ++i) {
                std::shared_ptr<ngraph::Node> result = context.function->get_output_op(i);
                std::shared_ptr<ngraph::Node> outputNode = result->get_input_node_shared_ptr(0);
                if (outputNode.get() == lastNode.get()) {
                    std::ostringstream oss;
                    oss << i;
                    originalNode->set_friendly_name(originalName + LayerTransformation::originalLayerPostfix);
                    lastNode->set_friendly_name(originalName + "." + oss.str());
                    break;
                }
            }
        }
    }
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
