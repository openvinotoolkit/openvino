// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/validation_util.hpp>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

atomic<size_t> Node::m_next_instance_id(0);

Node::Node(const Node& node)
    : m_control_dependents(node.m_control_dependents)
    , m_control_dependencies(node.m_control_dependencies)
    // skip m_node_type -- will be generated automatically
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_friendly_name(node.m_friendly_name)
    // skip m_unique_name -- will be generated automatically
    , m_provenance_tags(node.m_provenance_tags)
    , m_provenance_group(node.m_provenance_group)
    , m_inputs(node.m_inputs) // will be modified in the body
    // skip m_outputs -- should be initialized outside
    , m_op_annotations(node.m_op_annotations)
    , m_rt_info(node.m_rt_info)
{
    // cannot do it without copying node.m_inputs first due to too limiting const qualifiers
    for (auto& input : m_inputs)
    {
        input = descriptor::Input(this, input.get_index(), input.get_output());
        input.get_output().add_input(&input);
    }
}

Node& Node::operator=(const Node& node)
{
    this->m_control_dependents = node.m_control_dependents;
    this->m_control_dependencies = node.m_control_dependencies;
    this->m_instance_id = m_next_instance_id.fetch_add(1);
    this->m_friendly_name = node.m_friendly_name;
    this->m_provenance_tags = node.m_provenance_tags;
    this->m_provenance_group = node.m_provenance_group;
    this->m_inputs = node.m_inputs;
    this->m_op_annotations = node.m_op_annotations;
    this->m_rt_info = node.m_rt_info;
    // cannot do it without copying node.m_inputs first due to too limiting const qualifiers
    for (auto& input : m_inputs)
    {
        input = descriptor::Input(this, input.get_index(), input.get_output());
        input.get_output().add_input(&input);
    }
    return *this;
}

Node::Node(size_t output_size)
    : Node()
{
    set_output_size(output_size);
}

Node::Node(const OutputVector& arguments, size_t output_size)
    : Node()
{
    set_arguments(arguments);
    set_output_size(output_size);
}

Node::~Node()
{
    for (descriptor::Input& input : m_inputs)
    {
        if (input.has_output())
        {
            // This test adds 1 to the actual count, so a count of 2 means this input is the only
            // reference to the node.
            if (input.get_output().get_node().use_count() == 2)
            {
                // Don't want to trigger a deep recursive delete
                NodeVector nodes{input.get_output().get_node()};
                input.remove_output();
                safe_delete(nodes, true);
                return;
            }
            input.remove_output();
        }
    }
}

std::shared_ptr<Node> Node::copy_with_new_inputs(const OutputVector& inputs) const
{
    return copy_with_new_inputs(inputs, get_control_dependencies());
}

Output<const Node> Node::get_default_output() const
{
    return output(get_default_output_index());
}

Output<Node> Node::get_default_output()
{
    return output(get_default_output_index());
}

size_t Node::get_default_output_index() const
{
    return 0;
}

size_t Node::no_default_index() const
{
    NODE_VALIDATION_CHECK(this, false, "Default output not supported");
}

std::shared_ptr<Node>
    Node::copy_with_new_inputs(const OutputVector& inputs,
                               const std::vector<std::shared_ptr<Node>>& control_dependencies) const
{
    shared_ptr<Node> clone = clone_with_new_inputs(inputs);
    for (auto& cdep : control_dependencies)
    {
        clone->add_control_dependency(cdep);
    }
    for (size_t i = 0; i < get_output_size(); i++)
    {
        clone->get_output_tensor(i).set_names(get_output_tensor(i).get_names());
    }
    return clone;
}

void Node::safe_delete(NodeVector& nodes, bool recurse)
{
    for (auto& input : m_inputs)
    {
        if (input.has_output())
        {
            // This test adds 1 to the actual count, so a count of 2 means this input is the only
            // reference to the node.
            auto node = input.get_output().get_node();
            if (node.use_count() == 2)
            {
                // Move the node from the input to nodes so we don't trigger a deep recursive delete
                nodes.push_back(node);
            }
            input.remove_output();
        }
    }
    if (recurse)
    {
        while (nodes.size() > 0)
        {
            auto node = nodes.back();
            nodes.pop_back();
            node->safe_delete(nodes, false);
        }
    }
}

void Node::set_arguments(const NodeVector& arguments)
{
    OutputVector outputs;
    for (auto arg : arguments)
    {
        for (auto& output : arg->outputs())
        {
            outputs.push_back(output);
        }
    }
    set_arguments(outputs);
}

void Node::set_arguments(const OutputVector& arguments)
{
    // Add this node as a user of each argument.
    size_t i = 0;
    for (auto& output : arguments)
    {
        auto output_node = output.get_node();
        auto& output_descriptor = output_node->m_outputs.at(output.get_index());
        m_inputs.emplace_back(this, i++, output_descriptor);
    }
}

descriptor::Input& Node::get_input_descriptor(size_t position)
{
    while (m_inputs.size() <= position)
    {
        m_inputs.emplace_back(this, m_inputs.size());
    }
    return m_inputs.at(position);
}

descriptor::Output& Node::get_output_descriptor(size_t position)
{
    while (m_outputs.size() <= position)
    {
        size_t i = m_outputs.size();
        auto tensor_descriptor =
            make_shared<descriptor::Tensor>(element::dynamic, PartialShape::dynamic(), this, i);
        m_outputs.emplace_back(this, i, tensor_descriptor);
    }
    return m_outputs.at(position);
}

void Node::set_argument(size_t position, const Output<Node>& argument)
{
    auto output_node = argument.get_node();
    auto& output_descriptor = output_node->get_output_descriptor(argument.get_index());
    get_input_descriptor(position).replace_output(output_descriptor);
}

void Node::constructor_validate_and_infer_types()
{
    validate_and_infer_types();
}

void Node::set_output_size(size_t n)
{
    NGRAPH_CHECK(n >= m_outputs.size(), "shrinking ", m_outputs.size(), " to ", n);
    for (size_t i = m_outputs.size(); i < n; ++i)
    {
        // create the descriptors
        get_output_descriptor(i);
    }
}

void Node::invalidate_values()
{
    for (const auto& output : outputs())
        output.get_tensor().invalidate_values();
}

void Node::validate_and_infer_types() {}

void Node::set_input_is_relevant_to_shape(size_t i, bool relevant)
{
    NGRAPH_CHECK(i < m_inputs.size(),
                 "index '",
                 i,
                 "' out of range in set_input_is_relevant_to_shape(size_t index, bool relevant)");
    m_inputs[i].m_is_relevant_to_shape = relevant;
}

void Node::set_input_is_relevant_to_value(size_t i, bool relevant)
{
    NGRAPH_CHECK(i < m_inputs.size(),
                 "index '",
                 i,
                 "' out of range in set_input_is_relevant_to_value(size_t index, bool relevant)");
    m_inputs[i].m_is_relevant_to_value = relevant;
}

void Node::set_output_type(size_t i, const element::Type& element_type, const PartialShape& pshape)
{
    get_output_descriptor(i).get_tensor_ptr()->set_tensor_type(element_type, pshape);
}

std::string Node::description() const
{
    return get_type_name();
}

const std::string& Node::get_friendly_name() const
{
    if (m_friendly_name.empty())
    {
        return get_name();
    }
    return m_friendly_name;
}

const std::string& Node::get_name() const
{
    if (m_unique_name.empty())
    {
        const_cast<Node*>(this)->m_unique_name = description() + "_" + to_string(m_instance_id);
    }
    return m_unique_name;
}

void Node::set_friendly_name(const string& name)
{
    m_friendly_name = name;
}

void Node::add_provenance_group_member(const shared_ptr<Node>& node)
{
    m_provenance_group.insert(node);
}

void Node::remove_provenance_group_member(const shared_ptr<Node>& node)
{
    m_provenance_group.erase(node);
}

void Node::replace_provenance_group_member(const shared_ptr<Node>& current_node,
                                           const shared_ptr<Node>& replacement_node)
{
    // Catch up with the current state of the group
    replacement_node->add_provenance_tags(get_provenance_tags());
    if (current_node != nullptr)
    {
        remove_provenance_group_member(current_node);
        // Catch up with what was added to the current node
        replacement_node->add_provenance_tags(current_node->get_provenance_tags());
    }
    add_provenance_group_member(replacement_node);
}

const set<shared_ptr<Node>>& Node::get_provenance_group_members() const
{
    return m_provenance_group;
}

shared_ptr<Node> Node::add_provenance_group_members_above(const OutputVector& base)
{
    set<Node*> base_set;
    for (auto& output : base)
    {
        Node* node = output.get_node();
        if (node == this)
        {
            // A builder did nothing
            return shared_from_this();
        }
        base_set.insert(node);
    }
    vector<Node*> todo;
    for (auto value : input_values())
    {
        todo.push_back(value.get_node());
    }
    while (!todo.empty())
    {
        Node* node = todo.back();
        todo.pop_back();
        if (base_set.count(node) > 0)
        {
            continue;
        }
        add_provenance_group_member(node->shared_from_this());
        for (auto value : node->input_values())
        {
            if (m_provenance_group.count(value.get_node_shared_ptr()) == 0)
            {
                todo.push_back(value.get_node());
            }
        }
        base_set.insert(node);
    }
    return shared_from_this();
}

void Node::add_provenance_tags_above(const OutputVector& base,
                                     const std::unordered_set<std::string>& tag_set)
{
    set<Node*> base_set;
    for (auto& output : base)
    {
        base_set.insert(output.get_node());
    }
    vector<Node*> todo{this};
    while (!todo.empty())
    {
        Node* node = todo.back();
        todo.pop_back();
        if (base_set.count(node) > 0)
        {
            continue;
        }
        node->add_provenance_tags(tag_set);
        for (auto value : node->input_values())
        {
            todo.push_back(value.get_node());
        }
        base_set.insert(node);
    }
}

const std::unordered_set<std::string>& Node::get_provenance_tags() const
{
    return m_provenance_tags;
}

void Node::add_provenance_tag(const std::string& tag)
{
    m_provenance_tags.insert(tag);
    for (auto node : m_provenance_group)
    {
        node->add_provenance_tag(tag);
    }
}

void Node::remove_provenance_tag(const std::string& tag)
{
    m_provenance_tags.erase(tag);
}

void Node::merge_provenance_tags_from(const std::shared_ptr<const Node>& source)
{
    for (auto& tag : source->get_provenance_tags())
    {
        add_provenance_tag(tag);
    }
}

void Node::transfer_provenance_tags(const shared_ptr<Node>& replacement)
{
    auto common_args = ngraph::find_common_args(shared_from_this(), replacement);

    std::set<string> removed_subgraph_tags;

    auto set_replacement_prov = [&removed_subgraph_tags](std::shared_ptr<Node> node) {
        for (auto tag : node->get_provenance_tags())
        {
            removed_subgraph_tags.insert(tag);
        }
    };

    traverse_nodes({shared_from_this()}, set_replacement_prov, common_args);
    replacement->add_provenance_tags(removed_subgraph_tags);

    auto set_prov_new_nodes = [&removed_subgraph_tags](std::shared_ptr<Node> node) {
        node->add_provenance_tags(removed_subgraph_tags);
    };

    traverse_nodes({replacement}, set_prov_new_nodes, common_args);
}

Node* Node::get_input_node_ptr(size_t index) const
{
    NGRAPH_CHECK(
        index < m_inputs.size(), "index '", index, "' out of range in get_argument(size_t index)");
    return m_inputs[index].get_output().get_node().get();
}

std::shared_ptr<Node> Node::get_input_node_shared_ptr(size_t index) const
{
    NGRAPH_CHECK(
        index < m_inputs.size(), "index '", index, "' out of range in get_argument(size_t index)");
    return m_inputs[index].get_output().get_node();
}

Output<Node> Node::get_input_source_output(size_t i) const
{
    return input(i).get_source_output();
}

const std::vector<std::shared_ptr<Node>>& Node::get_control_dependencies() const
{
    return m_control_dependencies;
}

const std::vector<Node*>& Node::get_control_dependents() const
{
    return m_control_dependents;
}

void Node::add_control_dependency(std::shared_ptr<Node> node)
{
    if (find(m_control_dependencies.begin(), m_control_dependencies.end(), node) ==
        m_control_dependencies.end())
    {
        m_control_dependencies.push_back(node);
        if (find(node->m_control_dependents.begin(), node->m_control_dependents.end(), this) ==
            node->m_control_dependents.end())
        {
            node->m_control_dependents.push_back(this);
        }
    }
}

void Node::add_node_control_dependencies(std::shared_ptr<Node> source_node)
{
    for (auto& node : source_node->get_control_dependencies())
    {
        add_control_dependency(node);
    }
}

void Node::add_node_control_dependents(std::shared_ptr<Node> source_node)
{
    for (Node* node : source_node->get_control_dependents())
    {
        node->add_control_dependency(shared_from_this());
    }
}

void Node::transfer_control_dependents(std::shared_ptr<Node> replacement)
{
    replacement->add_node_control_dependents(shared_from_this());
    clear_control_dependents();
}

void Node::remove_control_dependency(std::shared_ptr<Node> node)
{
    {
        auto it = find(m_control_dependencies.begin(), m_control_dependencies.end(), node);
        if (it != m_control_dependencies.end())
        {
            m_control_dependencies.erase(it);
        }
    }
    {
        auto it = find(node->m_control_dependents.begin(), node->m_control_dependents.end(), this);
        if (it != node->m_control_dependents.end())
        {
            node->m_control_dependents.erase(it);
        }
    }
}

void Node::clear_control_dependencies()
{
    for (auto& node : m_control_dependencies)
    {
        auto it = find(node->m_control_dependents.begin(), node->m_control_dependents.end(), this);
        if (it != node->m_control_dependents.end())
        {
            node->m_control_dependents.erase(it);
        }
    }
    m_control_dependencies.clear();
}

void Node::clear_control_dependents()
{
    while (!m_control_dependents.empty())
    {
        (*m_control_dependents.begin())->remove_control_dependency(shared_from_this());
    }
}

const op::AutoBroadcastSpec& Node::get_autob() const
{
    static op::AutoBroadcastSpec s_spec;
    return s_spec;
}

namespace ngraph
{
    ostream& operator<<(ostream& out, const Node& node) { return node.write_description(out, 1); }
    ostream& operator<<(ostream& out, const Node* node) { return node->write_description(out, 1); }
} // namespace ngraph

std::ostream& Node::write_description(std::ostream& out, uint32_t depth) const
{
    if (depth == 0)
    {
        out << get_friendly_name();
    }
    else
    {
        out << "v" << get_type_info().version << "::" << get_type_info().name << " "
            << get_friendly_name() << " (";
        string sep = "";
        for (const auto& arg : input_values())
        {
            out << sep << arg;
            sep = ", ";
        }
        out << ") -> (";
        sep = "";
        for (size_t i = 0; i < get_output_size(); i++)
        {
            out << sep << get_output_element_type(i) << get_output_partial_shape(i);
            sep = ", ";
        }
        out << ")";
    }
    return out;
}

size_t Node::get_output_size() const
{
    return m_outputs.size();
}

const element::Type& Node::get_output_element_type(size_t i) const
{
    NGRAPH_CHECK(
        i < m_outputs.size(), "index '", i, "' out of range in get_output_element_type(size_t i)");
    return m_outputs[i].get_element_type();
}

const element::Type& Node::get_element_type() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error("get_element_type() must be called on a node with exactly one output.");
    }
    return get_output_element_type(0);
}

const Shape& Node::get_output_shape(size_t i) const
{
    NGRAPH_CHECK(
        i < m_outputs.size(), "index '", i, "' out of range in get_output_shape(size_t i)");
    return m_outputs[i].get_shape();
}

const PartialShape& Node::get_output_partial_shape(size_t i) const
{
    NGRAPH_CHECK(
        i < m_outputs.size(), "index '", i, "' out of range in get_output_partial_shape(size_t i)");
    return m_outputs[i].get_partial_shape();
}

const Shape& Node::get_shape() const
{
    if (get_output_size() != 1)
    {
        stringstream es;
        es << "get_shape() must be called on a node with exactly one output (" << description()
           << ")";
        throw ngraph_error(es);
    }
    return get_output_shape(0);
}

std::set<Input<Node>> Node::get_output_target_inputs(size_t i) const
{
    std::set<Input<Node>> result;

    for (auto& input : m_outputs.at(i).get_inputs())
    {
        result.emplace(input->get_raw_pointer_node(), input->get_index());
    }

    return result;
}

descriptor::Tensor& Node::get_output_tensor(size_t i) const
{
    NGRAPH_CHECK(
        i < m_outputs.size(), "index '", i, "' out of range in get_output_tensor(size_t i)");
    return m_outputs[i].get_tensor();
}

descriptor::Tensor& Node::get_input_tensor(size_t i) const
{
    NGRAPH_CHECK(i < m_inputs.size(), "index '", i, "' out of range in get_input_tensor(size_t i)");
    descriptor::Input input = m_inputs[i];
    return input.get_tensor();
}

size_t Node::get_input_size() const
{
    return m_inputs.size();
}

const element::Type& Node::get_input_element_type(size_t i) const
{
    NGRAPH_CHECK(
        i < m_inputs.size(), "index '", i, "' out of range in get_input_element_type(size_t i)");
    return m_inputs[i].get_element_type();
}

const Shape& Node::get_input_shape(size_t i) const
{
    NGRAPH_CHECK(i < m_inputs.size(), "index '", i, "' out of range in get_input_shape(size_t i)");
    return m_inputs[i].get_shape();
}

const PartialShape& Node::get_input_partial_shape(size_t i) const
{
    NGRAPH_CHECK(
        i < m_inputs.size(), "index '", i, "' out of range in get_input_partial_shape(size_t i)");
    return m_inputs[i].get_partial_shape();
}

NGRAPH_SUPPRESS_DEPRECATED_START
const string& Node::get_input_tensor_name(size_t i) const
{
    NGRAPH_CHECK(
        i < m_inputs.size(), "index '", i, "' out of range in get_input_tensor_name(size_t i)");
    return m_inputs[i].get_tensor().get_name();
}

const string& Node::get_output_tensor_name(size_t i) const
{
    NGRAPH_CHECK(
        i < m_outputs.size(), "index '", i, "' out of range in get_output_tensor_name(size_t i)");
    return m_outputs[i].get_tensor().get_name();
}
NGRAPH_SUPPRESS_DEPRECATED_END

bool Node::has_same_type(std::shared_ptr<const Node> node) const
{
    if (get_output_size() != node->get_output_size())
    {
        return false;
    }
    for (size_t i = 0; i < get_output_size(); ++i)
    {
        if (get_output_element_type(i) != node->get_output_element_type(i) ||
            get_output_shape(i) != node->get_output_shape(i))
        {
            return false;
        }
    }
    return true;
}

NodeVector Node::get_users(bool check_is_used) const
{
    NodeVector result;
    for (auto output : outputs())
    {
        for (auto input : output.get_target_inputs())
        {
            Node* input_node = input.get_node();
            if (!check_is_used || is_used(input_node))
            {
                result.push_back(input_node->shared_from_this());
            }
        }
    }
    return result;
}

std::string ngraph::node_validation_failure_loc_string(const Node* node)
{
    std::stringstream ss;
    ss << "While validating node '" << *node << "' with friendly_name '"
       << node->get_friendly_name() << '\'';
    return ss.str();
}

const std::shared_ptr<Node>& ngraph::check_single_output_arg(const std::shared_ptr<Node>& node,
                                                             size_t i)
{
    NGRAPH_CHECK(
        node->get_output_size() == 1, "Argument ", i, node, " must produce exactly one value.");
    return node;
}

const NodeVector& ngraph::check_single_output_args(const NodeVector& args)
{
    for (size_t i = 0; i < args.size(); ++i)
    {
        ngraph::check_single_output_arg(args.at(i), i);
    }
    return args;
}

OutputVector ngraph::as_output_vector(const NodeVector& args)
{
    OutputVector output_vector;
    for (auto arg : args)
    {
        output_vector.push_back(arg);
    }
    return output_vector;
}

NodeVector ngraph::as_node_vector(const OutputVector& values)
{
    NodeVector node_vector;
    for (auto& value : values)
    {
        node_vector.emplace_back(value.get_node_shared_ptr());
    }
    return node_vector;
}

ResultVector ngraph::as_result_vector(const OutputVector& values)
{
    ResultVector result;
    for (auto value : values)
    {
        shared_ptr<Node> node = value.get_node_shared_ptr();
        result.push_back(is_type<op::Result>(node) ? as_type_ptr<op::Result>(node)
                                                   : make_shared<op::Result>(value));
    }
    return result;
}

bool Node::match_value(pattern::Matcher* matcher,
                       const Output<Node>& pattern_value,
                       const Output<Node>& graph_value)
{
    if (pattern_value.get_index() != graph_value.get_index() ||
        (matcher->is_strict_mode() &&
         (!pattern_value.get_element_type().compatible(graph_value.get_element_type()) ||
          !pattern_value.get_partial_shape().compatible(graph_value.get_partial_shape()))))
    {
        return false;
    }
    return match_node(matcher, graph_value);
}

bool Node::match_node(pattern::Matcher* matcher, const Output<Node>& graph_value)
{
    matcher->add_node(graph_value);
    // Check if a type of a given node, which produces graph_value, matches the type of `this` node
    // or `this` node type is an ancestor of that node type. It is not the exact matching, types of
    // the nodes
    // may not match, but they are connected by the inheritance relation.
    // Not exact matching allows using base classes in the patterns and successfully matching such
    // patterns
    // with sub-graph of descent nodes types.
    if (graph_value.get_node_shared_ptr()->get_type_info().is_castable(get_type_info()) &&
        matcher->match_arguments(this, graph_value.get_node_shared_ptr()))
    {
        auto& pattern_map = matcher->get_pattern_value_map();
        pattern_map[shared_from_this()] = graph_value;
        return true;
    }
    return false;
}

// default implementation for the node to check if it contains partial shape
// we will override this method, for the Op's which depends on additional shape
// attribute to determine if node contains partial shape or not
bool Node::is_dynamic() const
{
    for (size_t i = 0; i < get_input_size(); i++)
    {
        if (get_input_partial_shape(i).is_dynamic())
        {
            return true;
        }
    }
    return false;
}

Input<Node> Node::input(size_t input_index)
{
    if (input_index >= m_inputs.size())
    {
        throw out_of_range("node input index is out of range");
    }

    return Input<Node>(this, input_index);
}

Output<Node> Node::input_value(size_t input_index) const
{
    return input(input_index).get_source_output();
}

Input<const Node> Node::input(size_t input_index) const
{
    if (input_index >= m_inputs.size())
    {
        throw out_of_range("node input index is out of range");
    }

    return Input<const Node>(this, input_index);
}

Output<Node> Node::output(size_t output_index)
{
    // All nodes will have at least 1 output
    if (output_index > 0 && output_index >= m_outputs.size())
    {
        throw out_of_range("node output index is out of range");
    }

    return Output<Node>(this, output_index);
}

Output<const Node> Node::output(size_t output_index) const
{
    // All nodes will have at least 1 output
    if (output_index > 0 && output_index >= m_outputs.size())
    {
        throw out_of_range("node output index is out of range");
    }

    return Output<const Node>(this, output_index);
}

vector<Input<Node>> Node::inputs()
{
    vector<Input<Node>> result;

    for (size_t i = 0; i < get_input_size(); i++)
    {
        result.emplace_back(this, i);
    }

    return result;
}

vector<Output<Node>> Node::input_values() const
{
    vector<Output<Node>> result;

    for (size_t i = 0; i < get_input_size(); i++)
    {
        result.emplace_back(input(i).get_source_output());
    }

    return result;
}

vector<Input<const Node>> Node::inputs() const
{
    vector<Input<const Node>> result;

    for (size_t i = 0; i < get_input_size(); i++)
    {
        result.emplace_back(this, i);
    }

    return result;
}

vector<Output<Node>> Node::outputs()
{
    vector<Output<Node>> result;

    for (size_t i = 0; i < get_output_size(); i++)
    {
        result.emplace_back(shared_from_this(), i);
    }

    return result;
}

vector<Output<const Node>> Node::outputs() const
{
    vector<Output<const Node>> result;

    for (size_t i = 0; i < get_output_size(); i++)
    {
        result.emplace_back(shared_from_this(), i);
    }

    return result;
}

bool Node::has_evaluate() const
{
    return false;
}

bool Node::evaluate(const HostTensorVector& output_values,
                    const HostTensorVector& input_values) const
{
    return false;
}

bool Node::evaluate(const HostTensorVector& output_values,
                    const HostTensorVector& input_values,
                    const EvaluationContext& evaluationContext) const
{
    return evaluate(output_values, input_values);
}

bool Node::evaluate_lower(const HostTensorVector& output_values) const
{
    const auto& inputs = input_values();
    bool dyn_inputs = std::any_of(inputs.begin(), inputs.end(), [](const Output<Node>& output) {
        return !output.get_tensor().has_and_set_bound();
    });
    if (dyn_inputs)
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool Node::evaluate_upper(const HostTensorVector& output_values) const
{
    const auto& inputs = input_values();
    bool dyn_inputs = std::any_of(inputs.begin(), inputs.end(), [](const Output<Node>& output) {
        return !output.get_tensor().has_and_set_bound();
    });
    if (dyn_inputs)
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool Node::constant_fold(OutputVector& output_values, const OutputVector& input_values)
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraph, "Node::constant_fold");

    if (m_rt_info.count("DISABLED_CONSTANT_FOLDING"))
    {
        return false;
    }

    // If all the inputs are constants, try to evaluate the outputs
    bool all_constants =
        std::all_of(input_values.begin(), input_values.end(), [](const Output<Node>& input) {
            return as_type_ptr<op::v0::Constant>(input.get_node_shared_ptr());
        });
    if (!all_constants)
        return false;

    HostTensorVector input_tensors;
    for (const auto& input : input_values)
    {
        auto host_tensor = make_shared<runtime::HostTensor>(
            as_type_ptr<op::v0::Constant>(input.get_node_shared_ptr()));
        input_tensors.push_back(host_tensor);
    }
    HostTensorVector output_tensors;
    OutputVector output_constants;
    for (const auto& output : outputs())
    {
        auto tensor =
            make_shared<HostTensor>(output.get_element_type(), output.get_partial_shape());
        output_tensors.push_back(tensor);
    }
    if (evaluate(output_tensors, input_tensors))
    {
        for (size_t i = 0; i < output_tensors.size(); ++i)
        {
            output_values[i] = make_shared<op::Constant>(output_tensors[i]);
        }
        return true;
    }
    return false;
}

constexpr DiscreteTypeInfo AttributeAdapter<shared_ptr<Node>>::type_info;

AttributeAdapter<std::shared_ptr<Node>>::AttributeAdapter(std::shared_ptr<Node>& value)
    : m_ref(value)
{
}

bool AttributeAdapter<std::shared_ptr<Node>>::visit_attributes(AttributeVisitor& visitor)
{
    auto original_id = visitor.get_registered_node_id(m_ref);
    auto id = original_id;
    visitor.on_attribute("ID", id);
    if (id != original_id)
    {
        m_ref = visitor.get_registered_node(id);
    }
    return true;
}

constexpr DiscreteTypeInfo AttributeAdapter<NodeVector>::type_info;

AttributeAdapter<NodeVector>::AttributeAdapter(NodeVector& ref)
    : m_ref(ref)
{
}

bool AttributeAdapter<NodeVector>::visit_attributes(AttributeVisitor& visitor)
{
    size_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size())
    {
        m_ref.resize(size);
    }
    ostringstream index;
    for (size_t i = 0; i < size; i++)
    {
        index.str("");
        index << i;
        string id;
        if (m_ref[i])
        {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i])
        {
            m_ref[i] = visitor.get_registered_node(id);
        }
    }
    return true;
}
