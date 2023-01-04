// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset5.hpp"

#include "vpu/utils/optional.hpp"
#include "vpu/ngraph/utilities.hpp"

#include "vpu/ngraph/transformations/extract_dynamic_batch/extract_dynamic_batch.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_mat_mul.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_convolution.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_binary_eltwise.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_unary_eltwise.hpp"

#include <queue>

namespace vpu {

ExtractBatch::ExtractBatch(std::unordered_set<ngraph::Node::type_info_t> targets) : targets(std::move(targets)) {}

namespace {

class Slicers {
public:
    static bool isSupported(const ngraph::Node& node) {
        return getSlicers().count(node.get_type_info());
    }

    static SliceConfiguration slice(const ngraph::Node& node) {
        const auto& slicers = getSlicers();
        const auto& type = node.get_type_info();
        return slicers.count(type) ? slicers.at(type)(node) : SliceConfiguration{};
    }

private:
    using Functor = std::function<SliceConfiguration(const ngraph::Node&)>;
    static const std::unordered_map<ngraph::DiscreteTypeInfo, Functor>& getSlicers() {
        static const std::unordered_map<ngraph::DiscreteTypeInfo, Functor>& slicers = {
            {ngraph::opset5::MatMul::get_type_info_static(),                  sliceMatMul},
            {ngraph::opset5::Convolution::get_type_info_static(),             sliceConvolution},
            {ngraph::opset5::GroupConvolution::get_type_info_static(),        sliceConvolution},
            {ngraph::opset5::ConvolutionBackpropData::get_type_info_static(), sliceConvolution},

            {ngraph::opset5::Add::get_type_info_static(),              sliceBinaryEltwise},
            {ngraph::opset5::Multiply::get_type_info_static(),         sliceBinaryEltwise},
            {ngraph::opset5::Minimum::get_type_info_static(),          sliceBinaryEltwise},
            {ngraph::opset5::Maximum::get_type_info_static(),          sliceBinaryEltwise},

            {ngraph::opset5::Relu::get_type_info_static(),             sliceUnaryEltwise},
            {ngraph::opset5::Clamp::get_type_info_static(),            sliceUnaryEltwise},

            // TODO: Need to make sure that all topologies/attributes scenarios can be covered by sliceUnaryEltwise
            {ngraph::opset5::MaxPool::get_type_info_static(),          sliceUnaryEltwise},
            {ngraph::opset5::AvgPool::get_type_info_static(),          sliceUnaryEltwise},
        };
        return slicers;
    }
};

struct SubGraph {
    Nodes leaves;
    Nodes all;
};

template<class Functor>
Nodes getNodes(ngraph::Node* from, ngraph::Node* to, Functor&& getNext) {
    auto visited = dfs(from, std::forward<Functor>(getNext), [to](ngraph::Node* node) { return node != to; });
    visited.erase(from);
    return visited;
}

template<class Functor>
SubGraph getLeaves(ngraph::Node* source, const Nodes& blackList, Functor&& getNext) {
    const auto isOk = [&blackList](ngraph::Node* node) { return Slicers::slice(*node).isSliceSupported() && !blackList.count(node); };
    Nodes leaves;
    auto visited = dfs(source, std::forward<Functor>(getNext), [isOk, getNext, &leaves](ngraph::Node* node) {
        const auto& nextNodes = getNext(node);
        const auto exit = nextNodes.empty() || std::any_of(nextNodes.cbegin(), nextNodes.cend(), [isOk](ngraph::Node* node) { return !isOk(node); });
        if (exit) {
            leaves.emplace(node);
            return false;
        }
        return true;
    });
    visited.erase(source);
    return {leaves, visited};
}

template<class NextForward, class NextBackward>
void getLeavesLCA(ngraph::Node* source, ngraph::Node*& lca, Nodes& nodes, const Nodes& leaves, const Nodes& allBackward,
                  NextForward&& getNextForward, NextBackward&& getNextBackward) {
    std::unordered_map<ngraph::Node*, std::size_t> depths{{ source, 0}}, leavesDepths;
    const auto less = [&depths](ngraph::Node* lhs, ngraph::Node* rhs) {
        VPU_THROW_UNLESS(depths.count(lhs), "There is no {} in all depth", lhs);
        VPU_THROW_UNLESS(depths.count(rhs), "There is no {} in all depth", rhs);
        return depths.at(lhs) < depths.at(rhs);
    };

    const auto equal = [&depths](ngraph::Node* lhs, ngraph::Node* rhs) {
        VPU_THROW_UNLESS(depths.count(lhs), "There is no {} in all depth", lhs);
        VPU_THROW_UNLESS(depths.count(rhs), "There is no {} in all depth", rhs);
        return depths.at(lhs) == depths.at(rhs);
    };

    Nodes visited;
    if (leaves.size() == 1 && leaves.count(source)) {
        lca = source;
        nodes = visited;
        return;
    }

    Nodes prevNodes;
    bfs(
        source,
        [getNextBackward, &allBackward, &prevNodes](const ngraph::Node* current) {
            prevNodes = getNextBackward(current);
            for (auto it = prevNodes.begin(); it != prevNodes.end();) {
                it = allBackward.count(*it) ? prevNodes.erase(it) : std::next(it);
            }
            return prevNodes.size();
        },
        [&](ngraph::Node* current) {
            if (current == source) {
                return true;
            }
            const auto depth = depths.at(*std::max_element(prevNodes.cbegin(), prevNodes.cend(), less)) + 1;
            depths[current] = depth;

            if (leaves.count(current)) {
                leavesDepths[current] = depth;
                return false;
            }
            return true;
        },
        [getNextForward](std::deque<ngraph::Node*>& deque, const ngraph::Node* current) {
            const auto& nextNodes = getNextForward(current);
            std::copy(nextNodes.cbegin(), nextNodes.cend(), std::back_inserter(deque));
        });

    VPU_THROW_UNLESS(leavesDepths.size() == leaves.size(), "leavesDepths and leaves have different sizes: {} vs {}", leavesDepths.size(), leaves.size());

    auto lcaCandidates = leaves;
    const auto minDepthArg = std::min_element(lcaCandidates.cbegin(), lcaCandidates.cend(), less);
    while (!std::all_of(lcaCandidates.cbegin(), lcaCandidates.cend(), [equal, minDepthArg](ngraph::Node* end) { return equal(end, *minDepthArg); })) {
        std::unordered_map<ngraph::Node*, ngraph::Node*> updates;
        for (const auto& end : lcaCandidates) {
            auto current = end;
            while (!equal(current, *minDepthArg)) {
                const auto& nextNodes = getNextBackward(current);
                current = *std::max_element(nextNodes.cbegin(), nextNodes.cend(), less);
            }

            updates[end] = current;
        }

        for (const auto& update : updates) {
            lcaCandidates.erase(update.first);
            lcaCandidates.emplace(update.second);
        }
    }

    while (lcaCandidates.size() != 1) {
        std::unordered_map<ngraph::Node*, ngraph::Node*> updates;
        for (const auto& end : lcaCandidates) {
            const auto& nextNodes = getNextBackward(end);
            const auto next = *std::max_element(nextNodes.cbegin(), nextNodes.cend(), less);

            updates[end] = next;
        }

        for (const auto& update : updates) {
            lcaCandidates.erase(update.first);
            lcaCandidates.emplace(update.second);
        }
    }

    lca = *lcaCandidates.begin();
    nodes = getNodes(source, lca, getNextForward);
}

template<class Functor>
std::shared_ptr<ngraph::opset5::Loop> makeLoop(ngraph::Node* root, ngraph::Node* leaf, Functor&& getNextTop) {
    ngraph::ParameterVector parameters;
    ngraph::ResultVector results;
    std::unordered_map<std::shared_ptr<ngraph::opset5::Parameter>, ngraph::Output<ngraph::Node>> slicedInputs, invariantInputs;
    std::set<ngraph::Output<ngraph::Node>> concatenatedResults;
    std::set<ngraph::Output<ngraph::Node>> iterValueResults;

    std::map<ngraph::Output<ngraph::Node>, ngraph::Output<ngraph::Node>> nodes;
    const auto getInput = [&nodes, &parameters, &slicedInputs, &invariantInputs](const ngraph::Output<ngraph::Node>& currentInput) {
        if (nodes.count(currentInput)) {
            return nodes.at(currentInput);
        } else {
            const auto& currentInputNode = currentInput.get_node();
            VPU_THROW_UNLESS(ngraph::op::is_constant(currentInputNode) || ngraph::op::is_parameter(currentInputNode),
                "Encountered intermediate node {} which is not cloned yet", currentInputNode);

            // assume if constant has several consumers all of them requires either Slice or Unchanged
            const auto& targetInputs = currentInput.get_target_inputs();
            const auto adjacentDiff = std::adjacent_find(targetInputs.cbegin(), targetInputs.cend(),
                [](const ngraph::Input<ngraph::Node>& lhs, const ngraph::Input<ngraph::Node>& rhs) {
                    const auto& lhsNode = lhs.get_node();
                    const auto& rhsNode = rhs.get_node();

                    const auto& lhsSplitConfig = Slicers::slice(*lhsNode);
                    const auto& rhsSplitConfig = Slicers::slice(*rhsNode);
                    if (!lhsSplitConfig.isSliceSupported() || !rhsSplitConfig.isSliceSupported()) {
                        return true;
                    }

                    const auto& lhsInputSplitConfig = lhsSplitConfig.inputs();
                    const auto& rhsInputSplitConfig = rhsSplitConfig.inputs();
                    return lhsInputSplitConfig[lhs.get_index()] != rhsInputSplitConfig[rhs.get_index()];
                });

            VPU_THROW_UNLESS(adjacentDiff == targetInputs.cend(),
                "Encountered constant {} that has 2 consumers ({} and {}) with different split configurations",
                currentInput, adjacentDiff->get_node(), std::next(adjacentDiff)->get_node());

            const auto& targetInput = targetInputs.begin();
            const auto& node = targetInput->get_node();
            const auto& index = targetInput->get_index();
            const auto splitInputConfiguration = Slicers::slice(*node).inputs();

            if (splitInputConfiguration[index] == SliceMode::Slice) {
                auto partialShape = currentInput.get_partial_shape();
                partialShape[0] = 1;
                auto parameter = std::make_shared<ngraph::opset5::Parameter>(currentInput.get_element_type(), partialShape);
                parameters.emplace_back(parameter);
                slicedInputs[parameter] = currentInput;

                nodes[currentInput] = parameter;
                return static_cast<ngraph::Output<ngraph::Node>>(parameter);
            } else {
                auto argument = currentInput;
                if (ngraph::op::is_parameter(currentInputNode)) {
                    auto parameter = std::make_shared<ngraph::opset5::Parameter>(currentInput.get_element_type(), currentInput.get_partial_shape());
                    parameters.emplace_back(parameter);
                    invariantInputs[parameter] = currentInput;

                    argument = parameter;
                }

                nodes[currentInput] = argument;
                return argument;
            }
        }
    };
    const auto clone = [getInput](const ngraph::Node* source) {
        std::vector<ngraph::Output<ngraph::Node>> newInputs;
        newInputs.reserve(source->get_input_size());
        const auto& currentInputs = source->input_values();
        std::transform(currentInputs.cbegin(), currentInputs.cend(), std::back_inserter(newInputs), getInput);

        auto cloned = source->clone_with_new_inputs(newInputs);
        cloned->set_friendly_name(source->get_friendly_name());
        VPU_THROW_UNLESS(cloned->get_output_size() == source->get_output_size(),
                         "Encountered mismatch in output count between original node {} and copy without batch {}", source, cloned);
        return cloned;
    };

    const auto splitInputConfiguration = Slicers::slice(*root).inputs();
    for (std::size_t i = 0; i < root->get_input_size(); ++i) {
        const auto& input = root->input_value(i);
        ngraph::Output<ngraph::Node> argument;
        if (splitInputConfiguration[i] == SliceMode::Slice) {
            auto partialShape = input.get_partial_shape();
            partialShape[0] = 1;

            auto parameter = std::make_shared<ngraph::opset5::Parameter>(input.get_element_type(), partialShape);
            parameters.emplace_back(parameter);
            slicedInputs[parameter] = input;

            argument = parameter;
        } else if (!ngraph::op::is_constant(input.get_node())) {
            auto parameter = std::make_shared<ngraph::opset5::Parameter>(input.get_element_type(), input.get_partial_shape());
            parameters.emplace_back(parameter);
            invariantInputs[parameter] = input;

            argument = parameter;
        } else {
            argument = input;
        }

        nodes[input] = argument;
    }

    std::shared_ptr<ngraph::Node> bodyNode;
    bfs(
        root,
        [getNextTop](const ngraph::Node* current) {
            return getNextTop(current).size();
        },
        [leaf, clone, &bodyNode](const ngraph::Node* current) {
            bodyNode = clone(current);
            return current != leaf;
        },
        [&](std::deque<ngraph::Node*>& deque, ngraph::Node* current) {
            for (std::size_t i = 0; i < current->get_output_size(); ++i) {
                const auto& currentOutput = current->output(i);
                const auto& bodyOutput = bodyNode->output(i);
                const auto& currentOutputNode = currentOutput.get_node();
                if (ngraph::op::is_output(currentOutputNode)) {
                    const auto splitOutputConfiguration = Slicers::slice(*current).outputs();
                    auto& outputCategory = splitOutputConfiguration[i] == SliceMode::Slice ? concatenatedResults : iterValueResults;
                    outputCategory.emplace(bodyOutput);
                    results.emplace_back(std::make_shared<ngraph::opset5::Result>(bodyOutput));
                } else {
                    nodes[currentOutput] = bodyOutput;
                    const auto& consumers = current->get_output_target_inputs(i);
                    std::transform(consumers.cbegin(), consumers.cend(), std::back_inserter(deque),
                        [](const ngraph::Input<ngraph::Node>& consumer) { return consumer.get_node(); });
                }
            }
        });

    const auto splitOutputConfiguration = Slicers::slice(*leaf).outputs();
    for (std::size_t i = 0; i < bodyNode->get_output_size(); ++i) {
        const auto& output = bodyNode->output(i);
        auto result = std::make_shared<ngraph::opset5::Result>(output);
        auto& outputCategory = splitOutputConfiguration[i] == SliceMode::Slice ? concatenatedResults : iterValueResults;
        outputCategory.emplace(output);
        results.emplace_back(result);
    }

    VPU_THROW_UNLESS(!slicedInputs.empty(), "Failed to find sliced inputs for loop in extract batch");
    const auto& slicedInput = slicedInputs.begin()->second;
    const auto shapeOf = std::make_shared<ngraph::opset5::ShapeOf>(slicedInput);

    // constant's shape has to be scalar (not empty) since if this constant has empty shape, so Gather will
    // have empty shape as well (Gather produces scalar). When this Gather will become ScatterElementsUpdate
    // argument ScatterElementsUpdate shape inference function will fail, since it requires indices and updates
    // to have exactly the same shape (indices rank must be the same as rank of data input which is 1D vector,
    // so its rank = 1 != 0)
    const auto constant = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);

    // TODO: check all other sliced inputs have the same batch?
    const auto batchSize = std::make_shared<ngraph::opset5::Gather>(shapeOf, constant, constant);

    const auto executionCondition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
    auto loop = std::make_shared<ngraph::opset5::Loop>(batchSize, executionCondition);

    const auto iterationCondition = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{}, true);
    results.emplace_back(std::make_shared<ngraph::opset5::Result>(iterationCondition));
    auto body = std::make_shared<ngraph::Function>(results, parameters, "body");
    loop->set_function(body);
    loop->set_special_body_ports({-1, static_cast<std::int64_t>(results.size()) - 1});
    for (const auto& entry : slicedInputs) {
        loop->set_sliced_input(entry.first, entry.second, 0, 1, 1, -1, 0);
    }

    for (const auto& entry : invariantInputs) {
        loop->set_invariant_input(entry.first, entry.second);
    }

    for (const auto& entry : iterValueResults) {
        loop->get_iter_value(entry, -1);
    }

    for (const auto& entry : concatenatedResults) {
        loop->get_concatenated_slices(entry, 0, 1, 1, -1, 0);
    }

    loop->validate_and_infer_types();
    return loop;
}

template<class Functor>
bool updateExternals(const ngraph::Node* source, const Nodes& allForward, const Nodes& allBackward, Nodes& externals, Functor&& getNextBackward) {
    bool updated = false;
    for (const auto& node : allForward) {
        const auto& nextNodes = getNextBackward(node);
        const auto hasExternalConnection = std::any_of(nextNodes.cbegin(), nextNodes.cend(), [source, &allForward, &allBackward](ngraph::Node* next) {
            return !allForward.count(next) && !allBackward.count(next) && next != source;
        });
        if (hasExternalConnection) {
            externals.emplace(node);
            updated = true;
        }
    }
    return updated;
}

template<class NextTop, class NextBottom>
bool removeExternalConnections(ngraph::Node* source, SubGraph& topSubGraph, SubGraph& bottomSubGraph, Nodes& topExternals, Nodes& bottomExternals,
                               NextTop&& getNextTop, NextBottom&& getNextBottom) {
    bool hasBeenUpdated = false;

    bool hasNewTopExternals = false;
    bool hasNewBottomExternals = false;
    do {
        hasNewTopExternals = updateExternals(source, topSubGraph.all, bottomSubGraph.all, topExternals, getNextBottom);
        if (hasNewTopExternals) {
            topSubGraph = getLeaves(source, topExternals, getNextTop);
            hasBeenUpdated = true;
        }

        hasNewBottomExternals = updateExternals(source, bottomSubGraph.all, topSubGraph.all, bottomExternals, getNextTop);
        if (hasNewBottomExternals) {
            bottomSubGraph = getLeaves(source, bottomExternals, getNextBottom);
            hasBeenUpdated = true;
        }
    } while (hasNewTopExternals || hasNewBottomExternals);

    return hasBeenUpdated;
}

}  // namespace

bool ExtractBatch::run_on_model(const std::shared_ptr<ngraph::Function>& functionPointer) {
    auto& function = *functionPointer;
    bool changed = false;

    Nodes sources;
    for (const auto& operation : function.get_ordered_ops()) {
        if (operation->is_dynamic() && targets.count(operation->get_type_info())) {
            sources.emplace(operation.get());
        }
    }

    auto getNextTop = [](const ngraph::Node* node) {
        Nodes nextNodes;
        for (std::size_t i = 0; i < node->get_input_size(); ++i) {
            const auto next = node->get_input_source_output(i).get_node();
            if (ngraph::op::is_constant(next) || ngraph::op::is_parameter(next)) {
                continue;
            }
            nextNodes.emplace(next);
        }
        return nextNodes;
    };

    auto getNextBottom = [](const ngraph::Node* node) {
        Nodes nextNodes;
        for (std::size_t i = 0; i < node->get_output_size(); ++i) {
            const auto consumers = node->get_output_target_inputs(i);
            for (const auto consumer : consumers) {
                const auto next = consumer.get_node();
                if (ngraph::op::is_output(next)) {
                    continue;
                }
                nextNodes.insert(next);
            }
        }
        return nextNodes;
    };

    for (auto currentSource = sources.begin(); currentSource != sources.end(); currentSource = sources.erase(currentSource)) {
        const auto& source = *currentSource;

        VPU_THROW_UNLESS(Slicers::isSupported(*source),
            "{} was requested as target operation type for batch extraction, but functor for this type is not provided.", source->get_type_info());

        if (!Slicers::slice(*source).isSliceSupported()) {
            continue;
        }

        Nodes topExternals, bottomExternals;

        auto topSubGraph = getLeaves(source, topExternals, getNextTop);
        auto bottomSubGraph = getLeaves(source, bottomExternals, getNextBottom);

        removeExternalConnections(source, topSubGraph, bottomSubGraph, topExternals, bottomExternals, getNextTop, getNextBottom);

        ngraph::Node* top = nullptr;
        ngraph::Node* bottom = nullptr;
        do {
            getLeavesLCA(source, top, topSubGraph.all, topSubGraph.leaves, bottomSubGraph.all, getNextTop, getNextBottom);
            getLeavesLCA(source, bottom, bottomSubGraph.all, bottomSubGraph.leaves, topSubGraph.all, getNextBottom, getNextTop);
        } while (removeExternalConnections(source, topSubGraph, bottomSubGraph, topExternals, bottomExternals, getNextTop, getNextBottom));

        for (const auto& node : topSubGraph.all) {
            if (sources.count(node)) {
                sources.erase(node);
            }
        }

        for (const auto& node : bottomSubGraph.all) {
            if (sources.count(node)) {
                sources.erase(node);
            }
        }

        auto loop = makeLoop(top, bottom, getNextTop);
        auto bottomNode = bottom->shared_from_this();
        loop->set_friendly_name(bottomNode->get_friendly_name());
        ngraph::replace_node(bottomNode, loop);
        function.validate_nodes_and_infer_types();
        changed = true;
    }

    return changed;
}

}  // namespace vpu
