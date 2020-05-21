//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <atomic>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_input.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/output_vector.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    template <typename NodeType>
    class Input;

    template <typename NodeType>
    class Output;

    class AttributeVisitor;
    class Variant;
    class Node;

    class Function;

    namespace runtime
    {
        class HostTensor;
    }
    using HostTensor = runtime::HostTensor;
    using HostTensorPtr = std::shared_ptr<HostTensor>;
    using HostTensorVector = std::vector<HostTensorPtr>;

    // Intermal, controls whether GetOutputElement nodes are elided
    // Defaults to being elided. Transformer should set to false if
    // it has passes that depend on GetOutputElement.
    NGRAPH_API void set_remove_goe(bool value)
        NGRAPH_DEPRECATED("Remove dependencies on GetOrderedOutput");
    NGRAPH_API bool get_remove_goe() NGRAPH_DEPRECATED("Remove dependencies on GetOrderedOutput");

    namespace op
    {
        struct AutoBroadcastSpec;

        namespace v0
        {
            class Result;
        }
    } // namespace op

    namespace pattern
    {
        class Matcher;
    }

    using ResultVector = std::vector<std::shared_ptr<op::v0::Result>>;

    namespace autodiff
    {
        class Adjoints;
    }

    NGRAPH_API
    std::string node_validation_failure_loc_string(const Node* node);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);
    NGRAPH_API
    const NodeVector& check_single_output_args(const NodeVector& args);

    const std::shared_ptr<Node>& check_single_output_arg(const std::shared_ptr<Node>& node,
                                                         size_t i);

    NGRAPH_API
    OutputVector as_output_vector(const NodeVector& args);
    NGRAPH_API
    NodeVector as_node_vector(const OutputVector& values);
    /// Returns a ResultVector referencing values.
    ResultVector as_result_vector(const OutputVector& values);

    /// Alias useful for cloning
    using NodeMap = std::unordered_map<ngraph::Node*, std::shared_ptr<ngraph::Node>>;

/// \brief Used in evaluator switch statement so that the case type and evaluate call
/// are guaranteed to have the types match.
///
/// Use this in an evaluate_*() function like this
///    switch (arg0->get_element_type())
///    {
///        TYPE_CASE(i8)(arg0, arg1, out, broadcast_spec); break;
///        TYPE_CASE(i16)(arg0, arg1, out, broadcast_spec); break;
///
/// Each TYPE_CASE statement expands like this:
///   case element::Type_t::a: rc = evaluate<element::Type_t::a>(arg0, arg1, out, broadcast_spec)
///
/// \note Don't forget to put a break after each statement or it will fall through and generate
/// a runtime error.

#define TYPE_CASE(a)                                                                               \
    case element::Type_t::a: rc = evaluate<element::Type_t::a>

    /// Nodes are the backbone of the graph of Value dataflow. Every node has
    /// zero or more nodes as arguments and one value, which is either a tensor
    /// or a (possibly empty) tuple of values.
    class NGRAPH_API Node : public std::enable_shared_from_this<Node>
    {
        // For access to generate_adjoints.
        friend class autodiff::Adjoints;

        // For access to m_outputs.
        friend class descriptor::Input;

        // For access to m_inputs and m_outputs.
        template <typename NodeType>
        friend class Input;

        // For access to m_outputs.
        template <typename NodeType>
        friend class Output;

    public:
        /// \brief Verifies that attributes and inputs are consistent and computes output shapes
        /// and element types. Must be implemented by concrete child classes so that it
        /// can be run any number of times.
        ///
        /// Throws if the node is invalid.
        virtual void validate_and_infer_types();

        // Called in constructors during transition
        void constructor_validate_and_infer_types();

        using type_info_t = DiscreteTypeInfo;

    protected:
        std::tuple<element::Type, PartialShape> validate_and_infer_elementwise_args(
            const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());
        void validate_and_infer_elementwise_arithmetic(
            const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());
        void validate_and_infer_elementwise_logical(
            const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());

        /// \brief Construct an unitialized Node
        Node() {}
        /// \brief Construct an unitialized Node
        /// \param output_size Number of outputs for this node
        Node(size_t output_size);

        /// \brief Constructor for Node subclasses that have metaclasses.
        /// \param arguments Output i will connect to input i
        /// \param output_size Number of outputs for this node
        Node(const OutputVector& arguments, size_t output_size = 1);

        // For back-compatibility
        virtual void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
            NGRAPH_DEPRECATED("use OutputVector version instead")
        {
            generate_adjoints(adjoints, as_output_vector(deltas));
        }
        virtual void generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas) {}
        /// \brief Moves nodes that would be deleted from inputs to nodes to avoid stack overflows
        /// on deep networks.
        void safe_delete(NodeVector& nodes, bool recurse);

    public:
        virtual ~Node();

        virtual bool visit_attributes(AttributeVisitor& visitor) { return false; }
        virtual bool is_unary_elementwise_arithmetic() const { return false; }
        virtual bool is_binary_elementwise_arithmetic() const { return false; }
        virtual bool is_binary_elementwise_comparison() const { return false; }
        virtual bool is_binary_elementwise_logical() const { return false; }
        /// \returns true if node supports autobroadcast operations
        virtual bool supports_auto_broadcast() const { return false; }
        /// \returns the autobroadcasr spec
        virtual const op::AutoBroadcastSpec& get_autob() const;
        /// \returns true if the node can decompose
        virtual bool supports_decompose() const { return false; }
        /// \brief Evaluates the op on input_values putting results in output_values
        /// \returns true if successful
        virtual bool evaluate(const HostTensorVector& output_values,
                              const HostTensorVector& input_values);
        virtual bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values);
        /// \brief Decomposes the FusedOp into a sub-graph consisting of core ngraph ops
        ///
        /// \return A vector of nodes comprising the sub-graph. The order of output
        ///         tensors must match the match output tensors of the FusedOp
        virtual NodeVector decompose_op() const { return NodeVector(); }
        /// Returns the NodeTypeInfo for the node's class.
        /// During transition to type_info, returns a dummy type_info for Node if the class
        /// has not been updated yet.
        virtual const type_info_t& get_type_info() const = 0;
        const char* get_type_name() const { return get_type_info().name; }
        /// Sets/replaces the arguments with new arguments.
        void set_arguments(const NodeVector& arguments);
        /// Sets/replaces the arguments with new arguments.
        void set_arguments(const OutputVector& arguments);
        /// Sets/replaces the arguments with new arguments.
        void set_argument(size_t position, const Output<Node>& argument);

        /// Sets the number of outputs
        void set_output_size(size_t output_size);

        void revalidate_and_infer_types() { validate_and_infer_types(); }
        // Called after transition
        void delayed_validate_and_infer_types();

        /// \brief Get the string name for the type of the node, such as `Add` or `Multiply`.
        ///        The class name, must not contain spaces as it is used for codegen.
        /// \returns A const reference to the node's type name
        virtual const std::string& description() const;
        /// \brief Get the unique name of the node.
        /// \returns A const reference to the node's unique name.
        const std::string& get_name() const;

        /// \brief Sets a friendly name for a node. This does not overwrite the unique name
        ///        of the node and is retrieved via get_friendly_name(). Used mainly for debugging.
        ///        The friendly name may be set exactly once.
        /// \param name is the friendly name to set
        void set_friendly_name(const std::string& name);

        /// \brief Gets the friendly name for a node. If no friendly name has been set via
        ///        set_friendly_name then the node's unique name is returned.
        /// \returns A const reference to the node's friendly name.
        const std::string& get_friendly_name() const;

        /// Return true if this has the same implementing class as node. This
        /// will be used by the pattern matcher when comparing a pattern
        /// graph against the graph.
        bool is_same_op_type(const std::shared_ptr<Node>& node) const
        {
            return get_type_info() == node->get_type_info();
        }

        /// \brief Marks an input as being relevant or irrelevant to the output shapes of this
        ///        node.
        /// \param i The index of the input to mark as relevant or irrelevant.
        /// \param relevant true if the input is relevant to output shapes, false otherwise.
        ///
        /// This is used by the shape specialization pass to know which nodes must be statically
        /// evaluated in order to complete shape specialization. (For example, the shape input of
        /// DynReshape must be evaluated statically in order for the output shape to be
        /// determined.) By default, all inputs are marked as shape-irrelevant. Overrides of
        /// validate_and_infer_types should call this function to mark shape-relevant inputs.
        // TODO(amprocte): should be protected
        void set_input_is_relevant_to_shape(size_t i, bool relevant = true);

        /// \brief Marks an input as being relevant or irrelevant to the output values of this
        ///        node.
        /// \param i The index of the input to mark as relevant or irrelevant.
        /// \param relevant true if the input is relevant to output values, false otherwise.
        ///
        /// This is used by the shape specialization pass to cut short evaluation in cases where
        /// an input value does not actually have any effect on the output value of the node. (As
        /// of this writing, the only example of this is ShapeOf.) By default, all inputs are
        /// marked as value-relevant. Overrides of validate_and_infer_types should call this
        /// function to mark value-irrelevant inputs.
        // TODO(amprocte): should be protected
        void set_input_is_relevant_to_value(size_t i, bool relevant = true);

        // TODO(amprocte): should this be protected?
        void set_output_type(size_t i,
                             const element::Type& element_type,
                             const PartialShape& pshape);

        virtual bool is_parameter() const { return false; }
        virtual bool is_output() const;
        virtual bool is_constant() const;
        virtual bool is_null() const { return false; }
        virtual bool is_op() const { return false; }
        virtual bool is_pattern() const { return false; }
        virtual bool is_commutative() const { return false; }
        virtual bool is_dynamic() const;
        virtual bool has_state() const { return false; }
        size_t get_instance_id() const { return m_instance_id; }
        /// \brief Writes a description of a node to a stream
        /// \param os The stream; should be returned
        /// \param depth How many levels of inputs to describe
        /// \returns The stream os
        virtual std::ostream& write_description(std::ostream& os, uint32_t depth = 0) const;

        std::deque<descriptor::Input>& get_inputs() NGRAPH_DEPRECATED("use inputs() instead")
        {
            return m_inputs;
        }
        const std::deque<descriptor::Input>& get_inputs() const
            NGRAPH_DEPRECATED("use inputs() instead")
        {
            return m_inputs;
        }
        std::deque<descriptor::Output>& get_outputs() NGRAPH_DEPRECATED("use outputs() instead");
        const std::deque<descriptor::Output>& get_outputs() const
            NGRAPH_DEPRECATED("use outputs() instead");

        /// Get control dependencies registered on the node
        const std::vector<std::shared_ptr<Node>>& get_control_dependencies() const;

        /// Get nodes dependent on this node
        const std::vector<Node*>& get_control_dependents() const;

        /// This node cannot execute until node executes
        void add_control_dependency(std::shared_ptr<Node> node);

        /// Remove the dependency of this node on node
        void remove_control_dependency(std::shared_ptr<Node> node);

        /// Remove all dependencies from this node
        void clear_control_dependencies();

        /// Remove this node as a dependency from all dependent nodes
        void clear_control_dependents();

        /// This node absorbs the control dependencies of source_node
        void add_node_control_dependencies(std::shared_ptr<Node> source_node);

        /// This node becomes a dependent of every node dependent on source_node
        void add_node_control_dependents(std::shared_ptr<Node> source_node);

        /// This node's control dependencies are replaced by replacement
        void transfer_control_dependents(std::shared_ptr<Node> replacement);

        /// Returns the number of outputs from the node.
        size_t get_output_size() const;

        /// Returns the element type for output i
        const element::Type& get_output_element_type(size_t i) const;

        /// Checks that there is exactly one output and returns its element type
        // TODO: deprecate in favor of node->get_output_element_type(0) with a suitable check in
        // the calling code, or updates to the calling code if it is making an invalid assumption
        // of only one output.
        const element::Type& get_element_type() const;

        /// Returns the shape for output i
        const Shape& get_output_shape(size_t i) const;

        /// Returns the partial shape for output i
        const PartialShape& get_output_partial_shape(size_t i) const;

        /// Second argument is ignored
        /// Returns the node if i=0 and the node has 1 output, otherwise a GetOutputElement
        /// If the node is a GetOutputElement, applies to the underlying node
        std::shared_ptr<Node> get_output_as_single_output_node(size_t i);

        /// Return the output to use when converting to an Output<Node> with no index specified.
        /// Throws when not supported.
        Output<const Node> get_default_output() const;
        Output<Node> get_default_output();

        /// Returns the output of the default output, or throws if there is none
        virtual size_t get_default_output_index() const;
        /// Throws no default
        size_t no_default_index() const;

        /// Checks that there is exactly one output and returns its shape
        // TODO: deprecate in favor of node->get_output_shape(0) with a suitable check in the
        // calling code, or updates to the calling code if it is making an invalid assumption of
        // only one output.
        const Shape& get_shape() const;

        /// Returns the tensor for output or input i
        descriptor::Tensor& get_output_tensor(size_t i) const;
        descriptor::Tensor& get_input_tensor(size_t i) const;

        /// Returns the tensor name for output i
        const std::string& get_output_tensor_name(size_t i) const;

        /// Checks that there is exactly one output and returns its tensor.
        descriptor::Tensor& get_output_tensor() const NGRAPH_DEPRECATED(
            "use node->get_output_tensor(0) instead; insert a check that the node has only one "
            "output, or update calling code not to assume only one output");

        /// Returns the tensor of output i
        // TODO: Investigate whether this really needs to be shared_ptr. If so, we'll need a
        // replacement in Output.
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr(size_t i) const
            NGRAPH_DEPRECATED("use &node->output(i).get_tensor() instead");

        /// Checks that there is exactly one output and returns its tensor.
        std::shared_ptr<descriptor::Tensor> get_output_tensor_ptr() const NGRAPH_DEPRECATED(
            "use &node->output(i).get_tensor() instead; insert a check that the node has only one "
            "output, or update calling code not to assume only one output");

        /// Returns the set of inputs using output i
        const std::vector<descriptor::Input*>& get_output_inputs(size_t i) const
            NGRAPH_DEPRECATED("use node->output(i).get_target_inputs() instead");

        std::set<Input<Node>> get_output_target_inputs(size_t i) const;

        /// Returns the number of inputs for the op
        size_t get_input_size() const;

        /// Returns the element type of input i
        // TODO: deprecate in favor of node->get_input_element_type(i)
        const element::Type& get_input_element_type(size_t i) const;

        /// Returns the shape of input i
        // TODO: deprecate in favor of node->get_input_shape(i)
        const Shape& get_input_shape(size_t i) const;

        /// Returns the partial shape of input i
        // TODO: deprecate in favor of node->get_input_partial_shape(i)
        const PartialShape& get_input_partial_shape(size_t i) const;

        /// Returns the tensor name for input i
        const std::string& get_input_tensor_name(size_t i) const;

        std::unordered_set<descriptor::Tensor*> liveness_new_list;
        std::unordered_set<descriptor::Tensor*> liveness_free_list;

        virtual NodeVector get_arguments() const NGRAPH_DEPRECATED("Use input_values().");
        std::shared_ptr<Node> get_argument(size_t index) const
            NGRAPH_DEPRECATED("use input_value(i).");

        Node* get_input_node_ptr(size_t index) const;
        std::shared_ptr<Node> get_input_node_shared_ptr(size_t index) const;
        Output<Node> get_input_source_output(size_t i) const;

    protected:
        // Will be replaced with clone_with_new_inputs
        virtual std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const
            NGRAPH_DEPRECATED("use copy_with_new_inputs instead");

    public:
        // TODO: When all copy_with_new_args have been replaced with copy_with_new_inputs, make
        // this pure and remove copy_with_new_args
        virtual std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const;

        std::shared_ptr<Node> copy_with_new_inputs(const OutputVector& new_args) const;

        std::shared_ptr<Node> copy_with_new_inputs(
            const OutputVector& inputs,
            const std::vector<std::shared_ptr<Node>>& control_dependencies) const;

        /// True if this and node have one output with same element type and shape
        bool has_same_type(std::shared_ptr<const Node> node) const;

        /// Get device placement
        Placement get_placement() const;

        /// Set device placement
        void set_placement(Placement placement);

        using RTMap = std::map<std::string, std::shared_ptr<Variant>>;

        RTMap& get_rt_info() { return m_rt_info; }
        const RTMap& get_rt_info() const { return m_rt_info; }
        const std::unordered_set<std::string>& get_provenance_tags() const;
        void add_provenance_tag(const std::string& tag);
        template <typename T>
        void add_provenance_tags(T tag_set)
        {
            for (auto tag : tag_set)
            {
                add_provenance_tag(tag);
            }
        }
        /// \brief Adds tag_set to this node and all intermediate nodes above base
        void add_provenance_tags_above(const OutputVector& base,
                                       const std::unordered_set<std::string>& tag_set);
        void remove_provenance_tag(const std::string& tag);
        /// \brief Add node to additional nodes that receive tags
        void add_provenance_group_member(const std::shared_ptr<Node>& node);
        /// \brief Remove node to additional nodes that receive tags
        void remove_provenance_group_member(const std::shared_ptr<Node>& node);
        /// \brief Replace current_node with replacement_node and transfer tags
        void replace_provenance_group_member(const std::shared_ptr<Node>& current_node,
                                             const std::shared_ptr<Node>& replacement_node);
        /// \return Provenance group nodes
        const std::set<std::shared_ptr<Node>>& get_provenance_group_members() const;

        /// \brief Add all nodes between this node and nodes in base as additional nodes to receive
        /// provenance tags.
        std::shared_ptr<Node> add_provenance_group_members_above(const OutputVector& base);

        // to be used when nodes are replaced
        void merge_provenance_tags_from(const std::shared_ptr<const Node>& source);

        /// Transfer provenance tags to replacement
        void transfer_provenance_tags(const std::shared_ptr<Node>& replacement);

        /// Get all the nodes that uses the current node
        NodeVector get_users(bool check_is_used = false) const;

        /// \return Version of this node
        virtual size_t get_version() const { return get_type_info().version; }
        virtual std::shared_ptr<Node> get_default_value() const { return nullptr; }
        /// Use instance ids for comparison instead of memory addresses to improve determinism
        bool operator<(const Node& other) const { return m_instance_id < other.m_instance_id; }
        /// \return A vector containing a handle for each of this node's inputs, in order.
        // TODO: Rename to get_inputs()?
        std::vector<Input<Node>> inputs();

        /// \return A vector containing a handle for each of this node's inputs, in order.
        std::vector<Input<const Node>> inputs() const;

        /// \return A vector containing the values for each input
        std::vector<Output<Node>> input_values() const;

        /// \return A vector containing a handle for each of this node's outputs, in order.
        // TODO: Rename to get_outputs()?
        std::vector<Output<Node>> outputs();

        /// \return A vector containing a handle for each of this node's outputs, in order.
        std::vector<Output<const Node>> outputs() const;

        /// \return A handle to the `input_index`th input of this node.
        /// \throw std::out_of_range if the node does not have at least `input_index+1` inputs.
        Input<Node> input(size_t input_index);

        /// \return A handle to the `input_index`th input of this node.
        /// \throw std::out_of_range if the node does not have at least `input_index+1` inputs.
        Input<const Node> input(size_t input_index) const;

        Output<Node> input_value(size_t input_index) const;

        /// \return A handle to the `output_index`th output of this node.
        /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
        Output<Node> output(size_t output_index);

        /// \return A handle to the `output_index`th output of this node.
        /// \throw std::out_of_range if the node does not have at least `output_index+1` outputs.
        Output<const Node> output(size_t output_index) const;

        void set_op_annotations(std::shared_ptr<ngraph::op::util::OpAnnotations> op_annotations)
        {
            m_op_annotations = op_annotations;
        }
        std::shared_ptr<ngraph::op::util::OpAnnotations> get_op_annotations() const
        {
            return m_op_annotations;
        }

        virtual bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value);

        virtual bool match_node(pattern::Matcher* matcher, const Output<Node>& graph_value);

    private:
        descriptor::Input& get_input_descriptor(size_t position);
        descriptor::Output& get_output_descriptor(size_t position);

        std::vector<Node*> m_control_dependents;
        std::vector<std::shared_ptr<Node>> m_control_dependencies;
        std::string m_node_type;
        size_t m_instance_id{m_next_instance_id.fetch_add(1)};
        std::string m_friendly_name;
        std::string m_unique_name;
        static std::atomic<size_t> m_next_instance_id;
        std::unordered_set<std::string> m_provenance_tags;
        std::set<std::shared_ptr<Node>> m_provenance_group;
        std::deque<descriptor::Input> m_inputs;
        std::deque<descriptor::Output> m_outputs;
        std::unordered_map<Node*, autodiff::Adjoints> m_adjoint_map;
        Placement m_placement = Placement::DEFAULT;
        std::shared_ptr<ngraph::op::util::OpAnnotations> m_op_annotations;
        std::map<std::string, std::shared_ptr<Variant>> m_rt_info;
    };

    using NodeTypeInfo = Node::type_info_t;

    NGRAPH_API std::ostream& operator<<(std::ostream&, const Node&);
    NGRAPH_API std::ostream& operator<<(std::ostream&, const Node*);

    // Like an Output but with a Node* instead of a shared_ptr<Node>
    struct RawNodeOutput
    {
        RawNodeOutput(const Output<Node>& value)
            : node(value.get_node())
            , index(value.get_index())
        {
        }
        RawNodeOutput(Node* node, size_t index)
            : node(node)
            , index(index)
        {
        }
        RawNodeOutput(const RawNodeOutput&) = default;
        RawNodeOutput() = default;
        RawNodeOutput& operator=(const RawNodeOutput&) = default;

        Node* node;
        size_t index{0};

        operator Output<Node>() { return Output<Node>(node->shared_from_this(), index); }
        bool operator==(const RawNodeOutput& other) const
        {
            return node == other.node && index == other.index;
        }
        bool operator!=(const RawNodeOutput& other) const { return !(*this == other); }
        bool operator<(const RawNodeOutput& other) const
        {
            return node < other.node || (node == other.node && index < other.index);
        }
        bool operator>(const RawNodeOutput& other) const
        {
            return node > other.node || (node == other.node && index > other.index);
        }
        bool operator<=(const RawNodeOutput& other) const { return !(*this > other); }
        bool operator>=(const RawNodeOutput& other) const { return !(*this < other); }
    };

    /// \brief Visits a reference to a node that has been registered with the visitor.
    template <>
    class NGRAPH_API AttributeAdapter<std::shared_ptr<Node>> : public VisitorAdapter
    {
    public:
        AttributeAdapter(std::shared_ptr<Node>& value);

        bool visit_attributes(AttributeVisitor& visitor) override;
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<std::shared_ptr<Node>>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    protected:
        std::shared_ptr<Node>& m_ref;
    };

    template <>
    class NGRAPH_API AttributeAdapter<NodeVector> : public VisitorAdapter
    {
    public:
        AttributeAdapter(NodeVector& ref);

        bool visit_attributes(AttributeVisitor& visitor) override;

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<NodeVector>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    protected:
        NodeVector& m_ref;
    };

    using RawNodeOutputMap = std::map<RawNodeOutput, Output<Node>>;

    class NGRAPH_API NodeValidationFailure : public CheckFailure
    {
    public:
        NodeValidationFailure(const CheckLocInfo& check_loc_info,
                              const Node* node,
                              const std::string& explanation)
            : CheckFailure(check_loc_info, node_validation_failure_loc_string(node), explanation)
        {
        }
    };
}
#define NODE_VALIDATION_CHECK(node, ...)                                                           \
    NGRAPH_CHECK_HELPER(::ngraph::NodeValidationFailure, (node), __VA_ARGS__)

namespace ngraph
{
    template <typename T>
    void check_new_args_count(const Node* node, T new_args)
    {
        NODE_VALIDATION_CHECK(node,
                              new_args.size() == node->get_arguments().size(),
                              "copy_with_new_args() expected ",
                              node->get_arguments().size(),
                              " argument",
                              (node->get_arguments().size() == 1 ? "" : "s"),
                              " but got ",
                              new_args.size());
    }

} // namespace ngraph
