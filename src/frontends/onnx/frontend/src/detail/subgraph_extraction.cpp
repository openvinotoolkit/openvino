// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_extraction.hpp"
#if defined(_MSC_VER)
#    pragma warning(push)
// Protobuf: conversion from 'XXX' to 'YYY', possible loss of data
#    pragma warning(disable : 4244)
#endif
#include <onnx/onnx_pb.h>

#include <functional>
#include <stack>

#include "openvino/frontend/exception.hpp"

using namespace ::ONNX_NAMESPACE;
using namespace ov::frontend::onnx;

enum class PortType { InputPort, OutputPort };

namespace {
void validate_node_index(const GraphProto& graph, const int node_idx) {
    FRONT_END_GENERAL_CHECK(node_idx >= 0 && node_idx < graph.node_size(),
                            "The specified node index is out of range of nodes in the original model(idx: ",
                            std::to_string(node_idx),
                            "; nodes count in the model: ",
                            std::to_string(graph.node_size()),
                            ")");
}

void validate_port_index(const GraphProto& graph, const int node_idx, const int port_idx, const PortType& port_type) {
    const int ports_number =
        (port_type == PortType::InputPort) ? graph.node(node_idx).input().size() : graph.node(node_idx).output().size();
    FRONT_END_GENERAL_CHECK(port_idx >= 0 && port_idx < ports_number,
                            "The specified node with index: ",
                            std::to_string(node_idx),
                            " has not ",
                            (port_type == PortType::InputPort) ? "input" : "output",
                            " port with index: ",
                            std::to_string(port_idx));
}

template <typename T>
std::function<bool(const T&)> name_equals(const std::string& name) {
    return [&name](const T& onnx_object) -> bool {
        return onnx_object.name() == name;
    };
}

const auto is_equal_to = +[](const std::string& other) {
    return [&](const std::string& s) {
        return s == other;
    };
};

/// \brief Checks if an item with name equal to "name" already exists in the specified
///        container. A container item is expected to have a name() method.
template <typename Container>
bool already_exists(const Container& items, const std::string& name) {
    using std::begin;
    using std::end;
    return std::any_of(begin(items), end(items), name_equals<typename Container::value_type>(name));
}

/// \brief Checks if a tensor with name "name" is produced by an input of the graph
bool is_graph_input(const GraphProto& graph, const std::string& name) {
    return already_exists(graph.input(), name);
}

/// \brief Checks if a tensor with name "name" is produced by an initializer of the graph
bool is_graph_initializer(const GraphProto& graph, const std::string& name) {
    return already_exists(graph.initializer(), name);
}

/// \brief Looks up the index of a node that produces a tensor "input_name". Used to traverse
///        the graph bottom-up. Starts from a node index "current_node_idx" because it operates
///        on a topologically sorted graph.
int find_source_node_idx(const GraphProto& graph, const int current_node_idx, const std::string& input_name) {
    // Some operators (e.g. Clip) have optional inputs
    if (input_name.empty())
        return -1;

    for (int i = current_node_idx - 1; i >= 0; --i) {
        const auto& outputs = graph.node(i).output();
        const auto output_found = std::any_of(std::begin(outputs), std::end(outputs), is_equal_to(input_name));

        if (output_found) {
            return i;
        }
    }

    OPENVINO_THROW("Source node not found in the graph for node: " + std::to_string(current_node_idx) +
                   " and input name: " + input_name);
}

/// \brief Looks up a descriptor for a given tensor name. This descriptor contains inferred
///        shape information which is required to create new inputs and outputs in the graph.
const ValueInfoProto find_tensor_descriptor(const GraphProto& graph, const std::string& tensor_name) {
    const auto it = std::find_if(std::begin(graph.value_info()),
                                 std::end(graph.value_info()),
                                 name_equals<ValueInfoProto>(tensor_name));

    if (it != std::end(graph.value_info())) {
        return *it;
    } else {
        // If tensor descriptor couldn't be found value info has to be specified
        // as fully dynamic:
        // - Fully dynamic shape
        // - Unknown data type
        auto dynamic_value_info = ValueInfoProto();
        dynamic_value_info.set_name(tensor_name);
        auto type = dynamic_value_info.mutable_type();
        auto tensor_type = type->mutable_tensor_type();
        tensor_type->set_elem_type(TensorProto_DataType::TensorProto_DataType_UNDEFINED);
        return dynamic_value_info;
    }
}

std::string get_input_tensor_name(const GraphProto& graph, const InputEdge& edge) {
    validate_node_index(graph, edge.m_node_idx);
    validate_port_index(graph, edge.m_node_idx, edge.m_port_idx, PortType::InputPort);

    return graph.node(edge.m_node_idx).input(edge.m_port_idx);
}

std::string get_output_tensor_name(const GraphProto& graph, const OutputEdge& edge) {
    validate_node_index(graph, edge.m_node_idx);
    validate_port_index(graph, edge.m_node_idx, edge.m_port_idx, PortType::OutputPort);

    return graph.node(edge.m_node_idx).output(edge.m_port_idx);
}

/// \brief Inserts a new input to the graph and removes an initializer that produced a tensor
///        specified by an input edge passed to this function.
void replace_initializer_with_new_input(GraphProto& graph, const InputEdge& edge) {
    const auto tensor_name = get_input_tensor_name(graph, edge);
    const auto it = std::find_if(std::begin(graph.initializer()),
                                 std::end(graph.initializer()),
                                 name_equals<TensorProto>(tensor_name));

    FRONT_END_GENERAL_CHECK(it != std::end(graph.initializer()),
                            "Could not find an initializer in the graph: '",
                            tensor_name);

    if (!already_exists(graph.input(), tensor_name)) {
        const auto& initializer = *it;
        auto& new_input = *(graph.add_input());

        auto& new_input_tensor_type = *(new_input.mutable_type()->mutable_tensor_type());
        new_input_tensor_type.set_elem_type(initializer.data_type());

        auto& new_input_shape = *(new_input_tensor_type.mutable_shape());
        for (const auto initializer_dim : initializer.dims()) {
            auto& new_dim = *(new_input_shape.add_dim());
            new_dim.set_dim_value(initializer_dim);
        }

        *(new_input.mutable_name()) = tensor_name;
    }

    graph.mutable_initializer()->erase(it);
}

/// \brief Inserts a new input to the graph and connects it to the node designated by an input
///        edge passed to this function.
/// \note  input_consumers is number of nodes which consume a new input
/// \return A new input edge (along with "true") if a new input was added to the graph,
///         false + the original edge otherwise.
std::pair<bool, std::string> append_new_graph_input(GraphProto& graph, const InputEdge& edge, int input_consumers) {
    const auto tensor_name = get_input_tensor_name(graph, edge);
    if (already_exists(graph.input(), tensor_name) && !is_graph_initializer(graph, tensor_name)) {
        // no need to append a new input if an edge points to an existing one in the model
        return {false, tensor_name};
    }

    auto& target_node = *(graph.mutable_node(edge.m_node_idx));
    FRONT_END_GENERAL_CHECK(edge.m_port_idx < target_node.input().size(),
                            "Input '",
                            edge.m_port_idx,
                            "' not found in the inputs of node ",
                            edge.m_node_idx,
                            ". Cannot append a new graph input to this node.");

    // if an edge is connected to an initializer, the initializer is removed and substituted
    // with an input
    if (is_graph_initializer(graph, tensor_name)) {
        replace_initializer_with_new_input(graph, edge);
        return {false, tensor_name};
    } else {
        std::string new_input_name;
        if (!edge.m_new_input_name.empty()) {
            new_input_name = edge.m_new_input_name;
            FRONT_END_GENERAL_CHECK(!already_exists(graph.input(), new_input_name),
                                    "New custom input name: ",
                                    new_input_name,
                                    " already exist in the graph");
        } else if (input_consumers > 1) {
            new_input_name = target_node.output(0) + "/placeholder_port_" + std::to_string(edge.m_port_idx);
        } else {
            new_input_name = tensor_name;
        }
        auto& new_input = *(graph.add_input());
        // copy the intermediate tensor properties to the newly created input
        new_input.MergeFrom(find_tensor_descriptor(graph, tensor_name));
        *(new_input.mutable_name()) = new_input_name;
        // attach the new graph input to the target node's input
        auto target_input = target_node.mutable_input(edge.m_port_idx);
        *target_input = new_input_name;
        return {true, new_input_name};
    }
}

/// \brief Adds new outputs to the ONNX graph for an edge specified by a user
/// The shape for this output is taken from a previously executed shape inference of the
/// original model.
void append_new_graph_output(GraphProto& graph, const OutputEdge& edge) {
    const auto tensor_name = get_output_tensor_name(graph, edge);
    auto& new_output = *(graph.add_output());
    // copy the intermediate tensor's properties to the newly created
    new_output.MergeFrom(find_tensor_descriptor(graph, tensor_name));
    *(new_output.mutable_name()) = tensor_name;
}

/// \brief Removes all items from a container except the ones whose names are in items_to_keep
///        It's intended to work with ONNX graph inputs, outputs and initializers only.
template <typename Container>
void discard_by_name(Container& all_items, const std::set<std::string>& items_to_keep) {
    static_assert(std::is_same<typename Container::value_type, ValueInfoProto>::value ||
                      std::is_same<typename Container::value_type, TensorProto>::value,
                  "Unsupported value type of the container");

    // The tested item can be discarded if its name is not found in the items_to_keep set
    const auto can_be_discarded = [&items_to_keep](const typename Container::value_type& item) {
        return items_to_keep.count(item.name()) == 0;
    };

    using std::begin;
    using std::end;

    // move the elements-to-discard to the end of the container
    const auto new_end = std::remove_if(begin(all_items), end(all_items), can_be_discarded);
    // erase all of the discarded elements past the new end of the container
    all_items.erase(new_end, end(all_items));
}

/// \brief Removes all nodes from a container keeping the ones whose index is in nodes_to_keep
template <typename Container>
void discard_nodes(Container& all_nodes, const std::set<int>& nodes_to_keep) {
    static_assert(std::is_same<typename Container::value_type, NodeProto>::value,
                  "Unsupported value type of the container");

    int idx = 0;
    const auto discard_node = [&idx, &nodes_to_keep](const typename Container::value_type&) {
        return nodes_to_keep.count(idx++) == 0;
    };

    using std::begin;
    using std::end;

    const auto new_end = std::remove_if(begin(all_nodes), end(all_nodes), discard_node);

    all_nodes.erase(new_end, end(all_nodes));
}
}  // namespace

/* -----------------------------------------------------------------------------------------------*/

SubgraphExtractor::SubgraphExtractor(GraphProto& graph) : m_onnx_graph(graph), m_node_inputs(graph.node_size()) {
    // gathers information about the graph - input edges of every node and number of "consumers"
    // of all tensors in the graph
    for (int i = 0; i < graph.node_size(); ++i) {
        for (const auto& node_input : graph.node(i).input()) {
            m_node_inputs[i].push_back(node_input);
            m_tensor_consumers[node_input] += 1;
        }
    }
}

void SubgraphExtractor::add_new_inputs(const std::vector<InputEdge>& new_inputs, const bool merge_inputs) {
    if (merge_inputs && new_inputs.size() > 1) {
        std::map<std::string, int> new_inputs_consumers;
        int index = 0;
        int input_consumers = static_cast<int>(new_inputs.size());

        // count all input edges
        for (const auto& input_edge : new_inputs) {
            const auto name = get_input_tensor_name(m_onnx_graph, input_edge);
            new_inputs_consumers[name] += 1;
        }

        // check if cutting will be performed for a Node (in that case set input_consumers to 0,
        // it will reuse Node name as new input name)
        for (const auto& input : new_inputs_consumers) {
            if (input.second == m_tensor_consumers[input.first] && input.second > 1) {
                input_consumers = 0;

                // get index of the current edge from new_inputs in order to pass it to append_new_graph_input()
                auto it = std::find_if(new_inputs.begin(), new_inputs.end(), [&](const InputEdge& input_edge) {
                    return get_input_tensor_name(m_onnx_graph, input_edge) == input.first;
                });
                index = static_cast<int>(std::distance(new_inputs.begin(), it));
            }
        }

        const auto new_input = append_new_graph_input(m_onnx_graph, new_inputs[index], input_consumers);

        // set input for all other incoming input edges
        for (auto it = new_inputs.begin() + 1; it != new_inputs.end(); ++it) {
            auto& target_node = *(m_onnx_graph.mutable_node(it->m_node_idx));
            auto target_input = target_node.mutable_input(it->m_port_idx);
            *target_input = new_input.second;
        }

        if (new_input.first) {
            // the original edge should be replaced with a new one in the helper multimap
            // this information will later be used during the subgraph extraction stage
            replace_input_edge(new_inputs[0], new_input.second);
        }
    } else {
        for (const auto& input_edge : new_inputs) {
            const auto tensor_name = get_input_tensor_name(m_onnx_graph, input_edge);

            // in case an edge is connected to a single node, a new graph input should be added
            // and connected to that node; the new edge is an edge between the node and new input
            const auto new_input = append_new_graph_input(m_onnx_graph, input_edge, m_tensor_consumers[tensor_name]);

            if (new_input.first) {
                // the original edge should be replaced with a new one in the helper multimap
                // this information will later be used during the subgraph extraction stage
                replace_input_edge(input_edge, new_input.second);
            }
        }
    }
}

void SubgraphExtractor::add_new_outputs(const std::vector<OutputEdge>& new_outputs) {
    for (const auto& output_edge : new_outputs) {
        validate_node_index(m_onnx_graph, output_edge.m_node_idx);

        append_new_graph_output(m_onnx_graph, output_edge);
    }
}

void SubgraphExtractor::replace_input_edge(const InputEdge& old_edge, const std::string& new_input_name) {
    // set a new name of an input indicated by the old_edge
    m_node_inputs.at(old_edge.m_node_idx).at(old_edge.m_port_idx) = new_input_name;
}

void SubgraphExtractor::extract_subgraph(std::vector<OutputEdge> subgraph_outputs) {
    // when the user doesn't specify any outputs, all outputs of the original graph should be kept
    if (subgraph_outputs.empty()) {
        subgraph_outputs = all_output_edges();
    }

    SubgraphComponents subgraph;

    for (const auto& output_edge : subgraph_outputs) {
        // for each output edge find the nodes, inputs and initializers that contribute to the value
        // produced by this output - "output contributors"
        // a sum of all contributors of all outputs is the target subgraph
        subgraph += discover_output_contributors(output_edge, subgraph);
    }

    // using the subgraph components collected above, modify the underlying GraphProto
    extract_subgraph_from_onnx_model(subgraph);
}

SubgraphExtractor::SubgraphComponents SubgraphExtractor::discover_output_contributors(
    const OutputEdge& output_edge,
    const SubgraphComponents& already_collected) const {
    SubgraphComponents output_contributors;
    const auto already_visited = [&already_collected, &output_contributors](const int node_index) {
        return already_collected.nodes.count(node_index) > 0 || output_contributors.nodes.count(node_index) > 0;
    };

    const auto tensor_name = get_output_tensor_name(m_onnx_graph, output_edge);
    output_contributors.outputs.insert(tensor_name);

    // reverse DFS graph traversal
    std::stack<int> nodes_to_visit;
    nodes_to_visit.push(output_edge.m_node_idx);

    while (!nodes_to_visit.empty()) {
        const auto n = nodes_to_visit.top();
        nodes_to_visit.pop();

        // if a node has already been visited, return early because it's already marked as
        // a node to keep in the final extracted subgraph
        if (already_visited(n)) {
            continue;
        }

        output_contributors.nodes.insert(n);

        // check if the visitor reached any of the graph inputs
        // and/or keep looking for more contributors further up in the graph

        // when an input or initializer is reached, the visitor stops the lookup
        const auto& n_inputs = m_node_inputs[n];
        for (auto& input_name : n_inputs) {
            if (is_graph_input(m_onnx_graph, input_name)) {
                output_contributors.inputs.insert(input_name);
                // when an initializer has a matching graph input
                if (is_graph_initializer(m_onnx_graph, input_name)) {
                    output_contributors.initializers.insert(input_name);
                }
            } else if (is_graph_initializer(m_onnx_graph, input_name)) {
                // when an initializer doesn't have a corresponding input
                output_contributors.initializers.insert(input_name);
            } else {
                // if an edge points to another node (source node) it should be visited
                // in one of the future iterations
                const auto node_idx = find_source_node_idx(m_onnx_graph, n, input_name);
                if (node_idx >= 0)
                    nodes_to_visit.push(node_idx);
            }
        }
    }

    return output_contributors;
}

void SubgraphExtractor::extract_subgraph_from_onnx_model(const SubgraphComponents& subgraph) {
    discard_by_name(*(m_onnx_graph.mutable_input()), subgraph.inputs);
    discard_by_name(*(m_onnx_graph.mutable_initializer()), subgraph.initializers);
    discard_by_name(*(m_onnx_graph.mutable_output()), subgraph.outputs);
    discard_nodes(*(m_onnx_graph.mutable_node()), subgraph.nodes);
}

std::vector<OutputEdge> SubgraphExtractor::all_output_edges() const {
    std::vector<OutputEdge> all_outputs;

    for (const auto& graph_output : m_onnx_graph.output()) {
        const auto node_index = find_source_node_idx(m_onnx_graph, m_onnx_graph.node_size(), graph_output.name());
        OPENVINO_ASSERT(node_index >= 0);
        const auto& node_outputs = m_onnx_graph.node(node_index).output();
        const auto output_port_it = std::find(std::begin(node_outputs), std::end(node_outputs), graph_output.name());
        all_outputs.emplace_back(node_index, output_port_it - std::begin(node_outputs));
    }

    return all_outputs;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
