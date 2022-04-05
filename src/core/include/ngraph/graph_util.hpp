// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov {
namespace op {
namespace v0 {
class Parameter;
class Result;
}  // namespace v0
}  // namespace op
}  // namespace ov
namespace ngraph {

namespace op {
namespace v0 {
using ov::op::v0::Parameter;
using ov::op::v0::Result;
}  // namespace v0
}  // namespace op

inline std::shared_ptr<ngraph::Function> clone_function(const ngraph::Function& func, ngraph::NodeMap& node_map) {
    return ov::clone_model(func, node_map);
}

inline std::shared_ptr<ngraph::Function> clone_function(const ngraph::Function& func) {
    return ov::clone_model(func);
}

using ov::compare_constants;
using ov::replace_node;
using ov::replace_node_update_name;
using ov::replace_nodes;
using ov::replace_output_update_name;
using ov::topological_sort;
using ov::traverse_nodes;

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
NodeVector find_common_args(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);

/// Topological sort of just nodes
template <typename T>
std::vector<std::shared_ptr<Node>> subgraph_topological_sort(T nodes) {
    std::stack<Node*, std::vector<Node*>> nodes_to_do;
    std::unordered_set<Node*> nodes_done;
    std::unordered_set<Node*> nodes_to_emit;
    std::vector<std::shared_ptr<Node>> result;

    for (auto& node : nodes) {
        nodes_to_emit.insert(node.get());
        nodes_to_do.push(node.get());
    }
    // NB: Some centos versions implement std::list::size() by counting elements
    size_t nodes_remaining = nodes_to_emit.size();
    while (nodes_to_do.size() > 0 && nodes_remaining > 0) {
        Node* node = nodes_to_do.top();
        if (nodes_done.count(node) == 0) {
            bool can_add = true;
            size_t arg_count = node->get_input_size();
            for (size_t i = 0; i < arg_count; ++i) {
                Node* dep = node->get_input_node_ptr(arg_count - i - 1);
                if (nodes_done.count(dep) == 0 && nodes_to_emit.count(node) != 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            for (auto& depptr : node->get_control_dependencies()) {
                Node* dep = depptr.get();
                if (nodes_done.count(dep) == 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            if (can_add) {
                if (nodes_to_emit.count(node) != 0) {
                    result.push_back(node->shared_from_this());
                    nodes_remaining--;
                }
                nodes_to_do.pop();
                nodes_done.insert(node);
            }
        }

        else {
            nodes_to_do.pop();
        }
    }
    return result;
}

template <typename T>
void validate_nodes_and_infer_types(const T& nodes) {
    for (auto& node : subgraph_topological_sort(nodes)) {
        node->revalidate_and_infer_types();
    }
}

// Check if all paths from X to a result go through Y
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool is_post_dominated(Node* X, Node* Y);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool is_equal_to_const_value(const std::string& const_value, const Output<Node>& reduce_constant);

// input nodes are cloned and returned
// NodeMap input may contain default node mapping i.e. pre-cloned nodes
// NodeMap output (by reference) fully maps input and cloned nodes
NGRAPH_API
std::vector<std::shared_ptr<ngraph::Node>> clone_nodes(const std::vector<std::shared_ptr<ngraph::Node>>& nodes,
                                                       NodeMap& node_map);

// input nodes are cloned and returned
// NodeMap input may contain default node mapping i.e. pre-cloned nodes
// NodeMap output (by reference) fully maps input and cloned nodes
NGRAPH_API
std::list<std::shared_ptr<ngraph::Node>> clone_nodes(const std::vector<std::shared_ptr<ngraph::Node>>& nodes,
                                                     RawNodeOutputMap& node_map);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
std::pair<std::shared_ptr<op::v0::Result>, std::shared_ptr<op::v0::Parameter>> insert_result_parameter_split(
    const std::shared_ptr<Node>& src_node,
    const std::shared_ptr<Node>& dst_node);

NGRAPH_API
void insert_new_node_between(const std::shared_ptr<Node>& src_node,
                             const std::shared_ptr<Node>& dst_node,
                             const std::shared_ptr<Node>& new_node);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
std::shared_ptr<Node> make_zero(const element::Type& element_type, const Shape& shape);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
std::shared_ptr<Node> make_constant_from_string(std::string val, const element::Type& element_type, const Shape& shape);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool is_zero(const Output<Node>& reduce_constant);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
NodeVector get_subgraph_outputs(const NodeVector& nodes,
                                const NodeVector& exclusions,
                                bool ignore_unused = false,
                                bool ignore_output_duplicates = true);

// Extract sub-graph computing the `results`. Stops backward traversal at either a Parameter
// node
// or a node that belongs to args
NGRAPH_API
NodeVector extract_subgraph(const NodeVector& results, const NodeVector& args);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool is_one(const Output<Node>& reduce_constant);

// Returns true if `node` is live in the graph i.e. a result op
// transitively uses this `node`
NGRAPH_API
bool is_used(Node* node);

// Returns count of `node` users that are still live in the graph
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
size_t get_user_count(Node* node);

// Return true if a node's user could potentially overwrite
// the output of this node with in-place kernels
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool possibly_overwritten(Node* node);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool is_strided(const Strides& strides);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool is_valid_rank(const std::shared_ptr<Node>& node, std::vector<size_t> valid_ranks);

NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
void plot_graph(std::shared_ptr<Function> f,
                const std::string& filename,
                std::function<void(const Node& node, std::vector<std::string>& attributes)> = nullptr);

/// \return A vector containing handles for each input of dst that is connected to an output
///         of `src`.
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
std::vector<Input<Node>> get_inputs_from(Node& src, Node& dst);
/// \return A vector containing a handle for each output of src that is connected to an input
///         of `dst`.
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
std::vector<Output<Node>> get_outputs_to(Node& src, Node& dst);

/// Checks the func for graph cycles starting from results going backwards, then from parameters
/// going forward.
/// It returns true if a cycle is found and the first cycle encountered.
NGRAPH_DEPRECATED("This method is deprecated and will be removed soon")
NGRAPH_API
bool check_for_cycles(const ngraph::Function* func, ngraph::NodeVector& cycle_nodes, bool& is_bkwd_cycle);
}  // namespace ngraph

using ngraph::replace_node;
using ngraph::replace_output_update_name;
