// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "editor_types.hpp"

namespace ONNX_NAMESPACE {
class GraphProto;
class NodeProto;
class ValueInfoProto;
}  // namespace ONNX_NAMESPACE

namespace ov {
namespace frontend {
namespace onnx {

using ::ONNX_NAMESPACE::GraphProto;

/// \brief Subgraph extraction helper structure
struct SubgraphExtractor {
    SubgraphExtractor(GraphProto& graph);

    /// \brief Adds new inputs to the graph and connects them to the nodes indicated by
    ///        the provided input edges.
    /// \param merge_inputs Flag indicating wheteher new inputs are generated for each input edge (false)
    ///                     or there is only one input created for provided edges (true)
    void add_new_inputs(const std::vector<InputEdge>& new_inputs, const bool merge_inputs = false);

    /// \brief Adds new outputs to the graph with the same name as the nodes pointed to
    ///        by the input edges "new_outputs".
    void add_new_outputs(const std::vector<OutputEdge>& new_outputs);

    /// \brief Extracts the final subgraph by traversing the original model bottom-up
    ///        starting at each of the provided output edges. The extracted subgraph
    ///        contains all previously added inputs and potentially a subset of original
    ///        model's inputs that contribute to the value calculated in the output tensors.
    ///        In the end the underlying GraphProto is modified and obsolete elements
    ///        are discarded after this method call has finished.
    ///
    /// \param subgraph_outputs A list of expected outputs of the extracted subgraph.
    void extract_subgraph(std::vector<OutputEdge> subgraph_outputs);

    /// \brief Represents a subgraph of an ONNX model by holding a subset of nodes, inputs,
    ///        outputs and initializers of the original graph. Objects of this struct can be
    ///        merged into other instances using the += operator to build a subgraph from
    ///        smaller clusters.
    struct SubgraphComponents {
        SubgraphComponents() = default;
        SubgraphComponents(const SubgraphComponents&) = delete;
        SubgraphComponents(SubgraphComponents&&) = default;
        SubgraphComponents& operator=(const SubgraphComponents&) = delete;
        SubgraphComponents& operator=(SubgraphComponents&&) = default;

        std::set<int> nodes;
        std::set<std::string> inputs;
        std::set<std::string> initializers;
        std::set<std::string> outputs;

        SubgraphComponents& operator+=(SubgraphComponents&& other) {
            nodes.insert(other.nodes.begin(), other.nodes.end());
            inputs.insert(other.inputs.begin(), other.inputs.end());
            initializers.insert(other.initializers.begin(), other.initializers.end());
            outputs.insert(other.outputs.begin(), other.outputs.end());
            return *this;
        }
    };

private:
    GraphProto& m_onnx_graph;

    // Graph traversal helper: the input names of each node
    std::vector<std::vector<std::string>> m_node_inputs;
    // Number of consumers of all tensors in the graph
    std::map<std::string, int> m_tensor_consumers;

    /// \brief Replace the name of an input indicated by the old_edge with a new one in the
    ///        helper struct.
    ///        This is used by the output contributors discovery.
    void replace_input_edge(const InputEdge& old_edge, const std::string& new_input_name);

    /// \brief Returns a list of edges of each outputs of the graph "m_onnx_graph"
    std::vector<OutputEdge> all_output_edges() const;

    /// \brief Traverses the graph bottom-up and collects all nodes, inputs and initializers
    ///        that contribute to an output designated by the provided output edge.
    ///        A sum of such SubgraphComponents objects forms a target extracted subgraph.
    SubgraphComponents discover_output_contributors(const OutputEdge& output_edge,
                                                    const SubgraphComponents& already_collected) const;

    /// \brief Modifies the underlying GraphProto object and discards all obsolete elements.
    ///
    /// \param subgraph An object describing the subgraph to be extracted (elems to be kept)
    void extract_subgraph_from_onnx_model(const SubgraphComponents& subgraph);
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
