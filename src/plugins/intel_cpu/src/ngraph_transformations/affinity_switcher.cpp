// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "affinity_switcher.hpp"

#include <transformations/serialize.hpp>

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

namespace {
using namespace ngraph;
Output<Node> getClonedSubgraph(Output<Node> parent,
                               const std::shared_ptr<ngraph::Node> end,
                               std::set<ov::Input<ov::Node>> consumers) {
    auto check_add = [](const std::shared_ptr<Node>& add) {
        return is_type<opset1::Add>(add) &&
              !is_type<opset1::Constant>(add->get_input_node_shared_ptr(0)) &&
              !is_type<opset1::Constant>(add->get_input_node_shared_ptr(1));
    };

    std::shared_ptr<Node> node;
    while (parent.get_node_shared_ptr() != end) {
        if (consumers.size() == 1) {
            node = consumers.begin()->get_node()->shared_from_this();
            if (check_add(node)) {
                break;
            }

            auto inputs = node->input_values();
            inputs[0] = parent;
            const auto new_node = node->clone_with_new_inputs(inputs);
            new_node->set_friendly_name("");
            parent = new_node->output(0);
            consumers = node->get_output_target_inputs(0);
            if (node == end) {
                break;
            }
        } else {
            auto next_node = consumers.begin()->get_node()->shared_from_this();
            while (!check_add(next_node)) {
                next_node = next_node->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            }

            OutputVector new_add_inputs;
            for (const auto& consumer : consumers) {
                const auto consumer_node = consumer.get_node()->shared_from_this();

                if (consumer_node == next_node) {
                    new_add_inputs.push_back(parent);
                } else {
                    auto inputs = consumer_node->input_values();
                    inputs[0] = parent;
                    const auto new_node = consumer_node->clone_with_new_inputs(inputs);
                    new_node->set_friendly_name("");

                    auto new_parent = new_node->output(0);
                    auto old_consumers = consumer.get_node()->output(0).get_target_inputs();

                    auto new_input = getClonedSubgraph(new_parent, next_node, old_consumers);
                    new_add_inputs.push_back(new_input);
                }
            }
            const auto new_node = next_node->clone_with_new_inputs(new_add_inputs);
            new_node->set_friendly_name("");
            parent = new_node->output(0);
            consumers = next_node->get_output_target_inputs(0);
        }
    }

    return parent;
}

Output<Node> clone(const std::shared_ptr<ngraph::Node> start, const std::shared_ptr<opset1::Split>& split,
    const std::shared_ptr<ngraph::Node> end, size_t idx, std::set<ov::Input<ov::Node>> consumers_original) {
    auto consumers = consumers_original;
    Output<Node> parent = split->output(idx);
    std::shared_ptr<Node> node;
    return getClonedSubgraph(parent, end, consumers);
}

// switch to image affinity
bool switchToImageAffinity(std::shared_ptr<ngraph::Node> start, std::shared_ptr<ngraph::Node> end) {
    const size_t batchSize = end->get_input_shape(0)[0];
    OutputVector newNodes;

    const auto consumers = start->get_output_target_inputs(0);

    const size_t batch_size = start->get_output_partial_shape(0)[0].get_length();
    const auto axis = opset1::Constant::create(element::i32, {}, { 0 });
    const auto split = std::make_shared<opset1::Split>(start, axis, batch_size);

    for (size_t i = 0; i < batchSize; ++i) {
        auto cloned = clone(start, split, end, i, consumers);
        newNodes.push_back(cloned);
    }

    const auto concat = std::make_shared<ngraph::opset1::Concat>(newNodes, 0);
    replace_node(end, concat);
    copy_runtime_info(end, concat);

    return true;
}

// switch to layer affinity
bool switchToLayerAffinity(std::shared_ptr<ngraph::Node> start, std::shared_ptr<ngraph::Node> end) {
    return true;
}
} // namespace

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::AffinitySwitcher, "AffinitySwitcher", 0);

bool MKLDNNPlugin::AffinitySwitcher::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool rewritten = false;

    std::shared_ptr<Node> start = f->get_parameters()[0];
    std::shared_ptr<Node> end;
    for (const auto& node : f->get_ordered_ops()) {
        for (const auto& input : node->input_values()) {
            const auto pShape = input.get_partial_shape();
            const auto rank = pShape.rank();
            if (rank.is_dynamic() || rank.get_length() == 0 || pShape[0].is_dynamic() || pShape[0].get_length() == 1)
                continue;

            if (transformation_callback(node)) {
                end = node;
            }
        }
    }
    if (start && end) {
        rewritten |= switchToImageAffinity(start, end);
    }
    return rewritten;
}
