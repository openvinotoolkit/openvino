// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "low_precision/split.hpp"

namespace ov {
namespace pass {
namespace low_precision {

SplitTransformation::SplitTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(SplitTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Split>({ pattern::wrap_type<ov::opset1::Multiply>(), pattern::wrap_type<ov::opset1::Constant>() });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool SplitTransformation::transform(ov::pass::pattern::Matcher& m) {
    if (!canBeTransformed(m.get_match_root())) {
        return false;
    }

    const auto split = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    const auto dequantization = NetworkHelper::getDequantization(split, defaultPrecisions);

    OutputVector inputs = split->input_values();
    inputs[0] = dequantization.data;

    const auto newSplit = split->clone_with_new_inputs(inputs);
    newSplit->set_friendly_name(split->get_friendly_name());
    ov::copy_runtime_info(split, newSplit);

    const int64_t axis = ov::as_type_ptr<ov::opset1::Constant>(split->get_input_node_shared_ptr(1))->cast_vector<int64_t>()[0];
    const size_t normalizedAxis = ov::util::try_normalize_axis(axis, split->get_input_partial_shape(0).rank(), *split);
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
            const auto subtract = NetworkHelper::makeDequantizationSubtract(parent, splitedSub[i]);
            copy_runtime_info({ newSplit, subtract }, subtract);
            parent = subtract;
        }

        const auto multiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(parent, splitedMul[i]);
        NetworkHelper::setOutDataPrecisionForTypeRelaxed(multiply, dequantization.multiply->get_output_element_type(0));
        copy_runtime_info({ newSplit, multiply }, multiply);

        lastNodes.push_back(multiply);
        replacement.push_back(multiply);
    }

    for (size_t i = 0ul; i < newSplit->get_output_size(); ++i) {
        for (auto input : split->output(i).get_target_inputs()) {
            input.replace_source_output(replacement[i]);
        }
    }

    // We do it to avoid dequantization propagation to the shapeOf subgraphs
    for (size_t i = 0; i < replacement.size(); ++i) {
        for (const auto& input : replacement[i].get_target_inputs()) {
            if (const auto shapeOf = as_type_ptr<ov::opset1::ShapeOf>(input.get_node()->shared_from_this())) {
                const auto newShapeOf = shapeOf->clone_with_new_inputs({ newSplit->output(i) });
                replace_node_update_name(shapeOf, newShapeOf);
            }
        }
    }

    updateOutputs(lastNodes, newSplit);

    OPENVINO_DEBUG("LPT: done: ", newSplit);
    return true;
}


void SplitTransformation::updateOutputs(
    std::vector<std::shared_ptr<ov::Node>> lastNodes,
    std::shared_ptr<ov::Node> originalNode) const {
    if (lastNodes.size() == 1ul) {
        updateOutput(lastNodes[0], originalNode);
    } else {
        const std::string originalName = originalNode->get_friendly_name();
        for (size_t i = 0; i < lastNodes.size(); ++i) {
            const auto lastNode = lastNodes[i];
            for (auto output : lastNodes[i]->outputs()) {
                for (auto input : output.get_target_inputs()) {
                    if (ov::is_type<ov::opset1::Result>(input.get_node())) {
                        originalNode->set_friendly_name(originalName + LayerTransformation::originalLayerPostfix);
                        lastNode->set_friendly_name(originalName + "." + std::to_string(i));
                        break;
                    }
                }
            }
        }
    }
}

bool SplitTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

bool SplitTransformation::canBeTransformed(const std::shared_ptr<Node>& layer) const {
    return !NetworkHelper::getDequantization(layer, defaultPrecisions).empty() && layer->get_input_partial_shape(0).rank().is_static();
}

} // namespace low_precision
} // namespace pass
} // namespace ov
