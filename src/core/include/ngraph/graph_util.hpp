// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

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
#include "ngraph/deprecated.hpp"
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

using ov::compare_constants;
using ov::replace_node;
using ov::replace_node_update_name;
using ov::replace_nodes;
using ov::replace_output_update_name;
using ov::topological_sort;
using ov::traverse_nodes;

using NodeMap = std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>;

NGRAPH_API_DEPRECATED
NGRAPH_API
ov::NodeVector find_common_args(std::shared_ptr<ov::Node> target, std::shared_ptr<ov::Node> replacement);

/// Topological sort of just nodes
template <typename T>
NGRAPH_API_DEPRECATED std::vector<std::shared_ptr<ov::Node>> subgraph_topological_sort(T nodes) {
    std::stack<ov::Node*, std::vector<ov::Node*>> nodes_to_do;
    std::unordered_set<ov::Node*> nodes_done;
    std::unordered_set<ov::Node*> nodes_to_emit;
    std::vector<std::shared_ptr<ov::Node>> result;

    for (auto& node : nodes) {
        nodes_to_emit.insert(node.get());
        nodes_to_do.push(node.get());
    }
    // NB: Some centos versions implement std::list::size() by counting elements
    size_t nodes_remaining = nodes_to_emit.size();
    while (nodes_to_do.size() > 0 && nodes_remaining > 0) {
        ov::Node* node = nodes_to_do.top();
        if (nodes_done.count(node) == 0) {
            bool can_add = true;
            size_t arg_count = node->get_input_size();
            for (size_t i = 0; i < arg_count; ++i) {
                ov::Node* dep = node->get_input_node_ptr(arg_count - i - 1);
                if (nodes_done.count(dep) == 0 && nodes_to_emit.count(node) != 0) {
                    can_add = false;
                    nodes_to_do.push(dep);
                }
            }
            for (auto& depptr : node->get_control_dependencies()) {
                ov::Node* dep = depptr.get();
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
NGRAPH_API_DEPRECATED void validate_nodes_and_infer_types(const T& nodes) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    for (auto& node : subgraph_topological_sort(nodes)) {
        node->revalidate_and_infer_types();
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

// Check if all paths from X to a result go through Y
NGRAPH_API_DEPRECATED
NGRAPH_API
bool is_post_dominated(ov::Node* X, ov::Node* Y);

NGRAPH_API_DEPRECATED
NGRAPH_API
bool is_equal_to_const_value(const std::string& const_value, const ov::Output<ov::Node>& reduce_constant);

// input nodes are cloned and returned
// NodeMap input may contain default node mapping i.e. pre-cloned nodes
// NodeMap output (by reference) fully maps input and cloned nodes
NGRAPH_API_DEPRECATED
NGRAPH_API
std::vector<std::shared_ptr<ov::Node>> clone_nodes(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                                   NodeMap& node_map);

// input nodes are cloned and returned
// NodeMap input may contain default node mapping i.e. pre-cloned nodes
// NodeMap output (by reference) fully maps input and cloned nodes
NGRAPH_API_DEPRECATED
NGRAPH_API
std::list<std::shared_ptr<ov::Node>> clone_nodes(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                                 ov::RawNodeOutputMap& node_map);

NGRAPH_API_DEPRECATED
NGRAPH_API
std::pair<std::shared_ptr<op::v0::Result>, std::shared_ptr<op::v0::Parameter>> insert_result_parameter_split(
    const std::shared_ptr<ov::Node>& src_node,
    const std::shared_ptr<ov::Node>& dst_node);

NGRAPH_API_DEPRECATED
NGRAPH_API
void insert_new_node_between(const std::shared_ptr<ov::Node>& src_node,
                             const std::shared_ptr<ov::Node>& dst_node,
                             const std::shared_ptr<ov::Node>& new_node);

NGRAPH_API_DEPRECATED
NGRAPH_API
std::shared_ptr<ov::Node> make_zero(const ov::element::Type& element_type, const ov::Shape& shape);

NGRAPH_API_DEPRECATED
NGRAPH_API
std::shared_ptr<ov::Node> make_constant_from_string(std::string val,
                                                    const ov::element::Type& element_type,
                                                    const ov::Shape& shape);

NGRAPH_API_DEPRECATED
NGRAPH_API
bool is_zero(const ov::Output<ov::Node>& reduce_constant);

NGRAPH_API_DEPRECATED
NGRAPH_API
ov::NodeVector get_subgraph_outputs(const ov::NodeVector& nodes,
                                    const ov::NodeVector& exclusions,
                                    bool ignore_unused = false,
                                    bool ignore_output_duplicates = true);

// Extract sub-graph computing the `results`. Stops backward traversal at either a Parameter
// node
// or a node that belongs to args
NGRAPH_API_DEPRECATED
NGRAPH_API
ov::NodeVector extract_subgraph(const ov::NodeVector& results, const ov::NodeVector& args);

NGRAPH_API_DEPRECATED
NGRAPH_API
bool is_one(const ov::Output<ov::Node>& reduce_constant);

// Returns true if `node` is live in the graph i.e. a result op
// transitively uses this `node`
NGRAPH_API_DEPRECATED
NGRAPH_API
bool is_used(ov::Node* node);

// Returns count of `node` users that are still live in the graph
NGRAPH_API_DEPRECATED
NGRAPH_API
size_t get_user_count(ov::Node* node);

NGRAPH_API_DEPRECATED
NGRAPH_API
bool is_strided(const ov::Strides& strides);

NGRAPH_API_DEPRECATED
NGRAPH_API
bool is_valid_rank(const std::shared_ptr<ov::Node>& node, std::vector<size_t> valid_ranks);

NGRAPH_API_DEPRECATED
NGRAPH_API
void plot_graph(std::shared_ptr<ov::Model> f,
                const std::string& filename,
                std::function<void(const ov::Node& node, std::vector<std::string>& attributes)> = nullptr);

/// \return A vector containing handles for each input of dst that is connected to an output
///         of `src`.
NGRAPH_API_DEPRECATED
NGRAPH_API
std::vector<ov::Input<ov::Node>> get_inputs_from(ov::Node& src, ov::Node& dst);
/// \return A vector containing a handle for each output of src that is connected to an input
///         of `dst`.
NGRAPH_API_DEPRECATED
NGRAPH_API
std::vector<ov::Output<ov::Node>> get_outputs_to(ov::Node& src, ov::Node& dst);

/// Checks the func for graph cycles starting from results going backwards, then from parameters
/// going forward.
/// It returns true if a cycle is found and the first cycle encountered.
NGRAPH_API_DEPRECATED
NGRAPH_API
bool check_for_cycles(const ov::Model* func, ov::NodeVector& cycle_nodes, bool& is_bkwd_cycle);
}  // namespace ngraph

using ngraph::replace_node;
using ngraph::replace_output_update_name;
