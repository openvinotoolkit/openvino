// Copyright (C) 2018-2021 Intel Corporation
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

namespace ngraph
{
    namespace descriptor
    {
        class Input;
        class Output;
    } // namespace descriptor

    namespace op
    {
        namespace v0
        {
            class Parameter;
        }
    } // namespace op

    NGRAPH_API
    void traverse_nodes(const std::shared_ptr<const Function> p,
                        std::function<void(std::shared_ptr<Node>)> f);

    NGRAPH_API
    void traverse_nodes(const Function* p, std::function<void(std::shared_ptr<Node>)> f);

    /// \brief Visit each node in a sub-graph of the entire graph
    /// \param subgraph_results The output nodes of the sub-graph
    /// \param f Function to execute at each node in the traversal
    /// \param subgraph_params Input nodes of the sub-graph (optional)
    ///
    /// Traverses a sub-graph starting from subgraph_results moving up
    /// towards parameter nodes. Traversal stops if it hits a node in
    /// subgraph_params.
    ///
    /// Most useful for finding parameters of a graph directly from the
    /// result nodes and not from function parameters or extracting a
    /// subgraph relevant to the computation of certain outputs
    NGRAPH_API
    void traverse_nodes(const NodeVector& subgraph_results,
                        std::function<void(std::shared_ptr<Node>)> f,
                        const NodeVector& subgraph_params = {});

    /// \brief Replace the node `target` with the node `replacement`, i.e.,
    ///        redirect all users and control dependencies of `target` to
    ///        `replacement`.
    ///
    /// \param target Node to be replaced.
    /// \param replacement Node to replace `target` with.
    /// \param output_order Vector determines order of replacement node's outputs.
    ///
    /// This is primarily used in graph-rewriting passes. For example, we
    /// might "fuse" two Concat operations as follows:
    ///
    /// (Step 0: Original graph)
    ///
    ///   A                       B
    ///   |                       |
    ///   v                       v
    /// N0[Concat, concatenation_axis=3]     C
    ///          |                           |
    ///          v                           v
    ///        N1[Concat, concatenation_axis=3]
    ///          |                |
    ///          v                v
    ///       some_user         another_user
    ///
    /// (Step 1: Construct replacement)
    ///
    ///    shared_ptr<Node> new_N1 = make_shared<op::Concat>({A,B,C},3);
    ///
    ///   A----------------------------------------.
    ///   |                                        |
    ///   |                       B----------------)--.
    ///   |                       |                |  |
    ///   v                       v                |  |
    /// N0[Concat, concatenation_axis=3]     C-----)--)--.
    ///          |                           |     |  |  |
    ///          v                           v     v  v  v
    ///        N1[Concat, concatenation_axis=3]   new_N1[Concat, concatenation_axis=3]
    ///          |                |
    ///          v                v
    ///       some_user         another_user
    ///
    /// (Step 2: Replace N1 with new_N1)
    ///
    ///    replace_node(N1, new_N1);
    ///
    ///   A----------------------------------------.
    ///   |                                        |
    ///   |                       B----------------)--.
    ///   |                       |                |  |
    ///   v                       v                |  |
    /// N0[Concat, concatenation_axis=3]     C-----)--)--.
    ///          |                           |     |  |  |
    ///          v                           v     v  v  v
    ///        N1[Concat, concatenation_axis=3]   new_N1[Concat, concatenation_axis=3]
    ///                                                  |                |
    ///                                                  v                v
    ///                                               some_user         another_user
    ///
    /// (Step 3: N0 and N1 are now dead, nodes will be freed)
    ///
    ///    [happens automatically, once all shared_ptrs to N1 are released]
    ///
    ///   A----------------------------------------.
    ///                                            |
    ///                           B----------------)--.
    ///                                            |  |
    ///                                            |  |
    ///                                      C-----)--)--.
    ///                                            |  |  |
    ///                                            v  v  v
    ///                                           new_N1[Concat, concatenation_axis=3]
    ///                                                  |                |
    ///                                                  v                v
    ///                                               some_user         another_user
    ///
    /// NOTE 1: replace_node is not type-safe (the graph is not revalidated).
    /// For example, the following is allowed, even if node `some_user`
    /// requires an input of shape 2x2:
    ///
    /// (Before)
    ///      A(shape=2x2)  B(shape=3x3)
    ///      |
    ///      v
    ///   some_user(requires 2x2 input)
    ///
    /// (After -- graph is now invalid)
    ///
    ///      replace_node(A, B);
    ///
    ///      A(shape=2x2)  B(shape=3x3)
    ///                    |
    ///                    v
    ///                 some_user(requires 2x2 input)
    ///
    /// NOTE 2: it is possible to insert a cycle into the graph with
    /// replace_node, resulting in an invalid graph. Care must be taken to
    /// avoid this. One common example is when you are attempting to insert a
    /// new node `M` "after"` a node `N`. For example, you might expect this
    /// to work:
    ///
    ///    shared_ptr<Node> M = make_shared<SomeUnaryOp>(N);
    ///    replace_node(M, N);
    ///
    /// The problem is that at replacement time, N itself is a user of M. So
    /// we end up introducing a cycle as follows:
    ///
    ///       N
    ///       |
    ///       v
    ///  other users...
    ///
    ///      |||
    ///      vvv
    ///
    ///       N------------>M
    ///       |
    ///       v
    ///  other users...
    ///
    ///      |||
    ///      vvv
    ///
    ///               .----.
    ///              |      |
    ///              |      |
    ///       N      `----->M
    ///                     |
    ///                     v
    ///                other users...
    ///
    /// To avoid the cycle, a valid way to perform the above desired insertion would be,
    ///
    ///        auto new_N = N->clone_with_new_inputs(N->input_values());
    ///        shared_ptr<Node> M = make_shared<SomeUnaryOp>(new_N);
    ///        replace_node(N, M);
    NGRAPH_API
    void replace_node(std::shared_ptr<Node> target,
                      std::shared_ptr<Node> replacement,
                      const std::vector<int64_t>& output_order);

    /// Replace target.outputs[i] with replacement_values[i] and transfer control dependents and
    /// provenance from target to the node(s) in replacement_values.
    NGRAPH_API
    void replace_node(const std::shared_ptr<Node>& target, const OutputVector& replacement_values);

    NGRAPH_API
    void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);

    /// \brief Replace multiple nodes in a function.
    /// \param f Function where replacement is taking place.
    /// \param parameter_replacement_map A mapping from parameter shared pointers to parameter
    ///                                  shared pointers. For each pair (k,v) in the map, parameter
    ///                                  k is replaced by parameter v, except if k==v or k is not a
    ///                                  parameter bound by f, in which case the pair (k,v) is
    ///                                  ignored.
    /// \param body_replacement_map A mapping from node shared pointers to node shared pointers.
    ///                             For each pair (k,v) in the map, node k is replaced by node v,
    ///                             except if k==v, the pair (k,v) is ignored.
    ///                             Note that if k is a parameter, its users will be redirected to
    ///                             v, but k will _not_ be replaced in the function's parameter
    ///                             list.
    ///
    /// Limitations:
    ///
    ///    - No check is made that the replaced nodes in `parameter_replacement_map` are actually
    ///      among the bound parameters of `f`. (If a parameter appears in the map that is not
    ///      bound by `f`, it will be silently ignored.)
    ///    - If a parameter node appears as a key in both `parameter_replacement_map` _and_ in
    ///      `body_replacement_map`, behavior is unspecified.
    NGRAPH_API
    void replace_nodes(
        const std::shared_ptr<Function>& f,
        const std::unordered_map<std::shared_ptr<op::v0::Parameter>,
                                 std::shared_ptr<op::v0::Parameter>>& parameter_replacement_map,
        const std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>>&
            body_replacement_map);

    NGRAPH_API
    NodeVector find_common_args(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);

    /// Topological sort of nodes needed to compute root_nodes
    template <typename T>
    std::vector<std::shared_ptr<Node>> topological_sort(T root_nodes)
    {
        std::stack<Node*, std::vector<Node*>> nodes_to_do;
        std::unordered_set<Node*> nodes_done;
        std::vector<std::shared_ptr<Node>> result;

        for (auto& node : root_nodes)
        {
            nodes_to_do.push(node.get());
        }
        while (nodes_to_do.size() > 0)
        {
            Node* node = nodes_to_do.top();
            if (nodes_done.count(node) == 0)
            {
                bool can_add = true;
                size_t arg_count = node->get_input_size();
                for (size_t i = 0; i < arg_count; ++i)
                {
                    Node* dep = node->get_input_node_ptr(arg_count - i - 1);
                    if (nodes_done.count(dep) == 0)
                    {
                        can_add = false;
                        nodes_to_do.push(dep);
                    }
                }
                for (auto& depptr : node->get_control_dependencies())
                {
                    Node* dep = depptr.get();
                    if (nodes_done.count(dep) == 0)
                    {
                        can_add = false;
                        nodes_to_do.push(dep);
                    }
                }
                if (can_add)
                {
                    result.push_back(node->shared_from_this());
                    nodes_to_do.pop();
                    nodes_done.insert(node);
                }
            }
            else
            {
                nodes_to_do.pop();
            }
        }
        return result;
    }

    /// Topological sort of just nodes
    template <typename T>
    std::vector<std::shared_ptr<Node>> subgraph_topological_sort(T nodes)
    {
        std::stack<Node*, std::vector<Node*>> nodes_to_do;
        std::unordered_set<Node*> nodes_done;
        std::unordered_set<Node*> nodes_to_emit;
        std::vector<std::shared_ptr<Node>> result;

        for (auto& node : nodes)
        {
            nodes_to_emit.insert(node.get());
            nodes_to_do.push(node.get());
        }
        // NB: Some centos versions implement std::list::size() by counting elements
        size_t nodes_remaining = nodes_to_emit.size();
        while (nodes_to_do.size() > 0 && nodes_remaining > 0)
        {
            Node* node = nodes_to_do.top();
            if (nodes_done.count(node) == 0)
            {
                bool can_add = true;
                size_t arg_count = node->get_input_size();
                for (size_t i = 0; i < arg_count; ++i)
                {
                    Node* dep = node->get_input_node_ptr(arg_count - i - 1);
                    if (nodes_done.count(dep) == 0 && nodes_to_emit.count(node) != 0)
                    {
                        can_add = false;
                        nodes_to_do.push(dep);
                    }
                }
                for (auto& depptr : node->get_control_dependencies())
                {
                    Node* dep = depptr.get();
                    if (nodes_done.count(dep) == 0)
                    {
                        can_add = false;
                        nodes_to_do.push(dep);
                    }
                }
                if (can_add)
                {
                    if (nodes_to_emit.count(node) != 0)
                    {
                        result.push_back(node->shared_from_this());
                        nodes_remaining--;
                    }
                    nodes_to_do.pop();
                    nodes_done.insert(node);
                }
            }

            else
            {
                nodes_to_do.pop();
            }
        }
        return result;
    }

    template <typename T>
    void validate_nodes_and_infer_types(const T& nodes)
    {
        for (auto& node : subgraph_topological_sort(nodes))
        {
            node->revalidate_and_infer_types();
        }
    }

    // Check if all paths from X to a result go through Y
    NGRAPH_API
    bool is_post_dominated(Node* X, Node* Y);

    NGRAPH_API
    bool is_equal_to_const_value(std::string const_value, const Output<Node>& reduce_constant);

    // input nodes are cloned and returned
    // NodeMap input may contain default node mapping i.e. pre-cloned nodes
    // NodeMap output (by reference) fully maps input and cloned nodes
    NGRAPH_API
    std::vector<std::shared_ptr<ngraph::Node>>
        clone_nodes(const std::vector<std::shared_ptr<ngraph::Node>>& nodes, NodeMap& node_map);

    // input nodes are cloned and returned
    // NodeMap input may contain default node mapping i.e. pre-cloned nodes
    // NodeMap output (by reference) fully maps input and cloned nodes
    NGRAPH_API
    std::list<std::shared_ptr<ngraph::Node>>
        clone_nodes(const std::vector<std::shared_ptr<ngraph::Node>>& nodes,
                    RawNodeOutputMap& node_map);

    // input function is cloned and returned
    // NodeMap input may contain default node mapping i.e. pre-cloned nodes
    // NodeMap output (by reference) fully maps input and cloned function ops
    NGRAPH_API
    std::shared_ptr<ngraph::Function> clone_function(const ngraph::Function& func,
                                                     NodeMap& node_map);

    // input function is cloned and returned
    NGRAPH_API
    std::shared_ptr<ngraph::Function> clone_function(const ngraph::Function& func);

    NGRAPH_API
    std::pair<std::shared_ptr<op::Result>, std::shared_ptr<op::v0::Parameter>>
        insert_result_parameter_split(const std::shared_ptr<Node>& src_node,
                                      const std::shared_ptr<Node>& dst_node);

    NGRAPH_API
    void insert_new_node_between(const std::shared_ptr<Node>& src_node,
                                 const std::shared_ptr<Node>& dst_node,
                                 const std::shared_ptr<Node>& new_node);

    NGRAPH_API
    std::shared_ptr<Node> make_zero(const element::Type& element_type, const Shape& shape);

    NGRAPH_API
    std::shared_ptr<Node> make_constant_from_string(std::string val,
                                                    const element::Type& element_type,
                                                    const Shape& shape);

    NGRAPH_API
    bool is_zero(const Output<Node>& reduce_constant);

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

    NGRAPH_API
    bool is_one(const Output<Node>& reduce_constant);

    NGRAPH_API
    bool compare_constants(const std::shared_ptr<Node>& n1, const std::shared_ptr<Node>& n2);

    // Returns true if `node` is live in the graph i.e. a result op
    // transitively uses this `node`
    NGRAPH_API
    bool is_used(Node* node);

    // Returns count of `node` users that are still live in the graph
    NGRAPH_API
    size_t get_user_count(Node* node);

    // Return true if a node's user could potentially overwrite
    // the output of this node with in-place kernels
    NGRAPH_API
    bool possibly_overwritten(Node* node);

    NGRAPH_API
    bool is_strided(const Strides& strides);

    NGRAPH_API
    bool is_valid_rank(const std::shared_ptr<Node>& node, std::vector<size_t> valid_ranks);

    NGRAPH_API
    void plot_graph(
        std::shared_ptr<Function> f,
        const std::string& filename,
        std::function<void(const Node& node, std::vector<std::string>& attributes)> = nullptr);

    /// \return A vector containing handles for each input of dst that is connected to an output
    ///         of `src`.
    NGRAPH_API
    std::vector<Input<Node>> get_inputs_from(Node& src, Node& dst);
    /// \return A vector containing a handle for each output of src that is connected to an input
    ///         of `dst`.
    NGRAPH_API
    std::vector<Output<Node>> get_outputs_to(Node& src, Node& dst);

    /// Checks the func for graph cycles starting from results going backwards, then from parameters
    /// going forward.
    /// It returns true if a cycle is found and the first cycle encountered.
    NGRAPH_API
    bool check_for_cycles(const ngraph::Function* func,
                          ngraph::NodeVector& cycle_nodes,
                          bool& is_bkwd_cycle);

    NGRAPH_API
    bool replace_output_update_name(Output<Node> node, const Output<Node>& node_input);

    NGRAPH_API
    bool replace_node_update_name(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);
} // namespace ngraph
