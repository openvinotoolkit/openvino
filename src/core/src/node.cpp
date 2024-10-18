// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"

#include <memory>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include "atomic_guard.hpp"
#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/descriptor/input.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "shape_validation.hpp"
#include "shared_node_info.hpp"

using namespace std;

namespace {
static const char node_idx_out_of_range_txt[] = "node index is out of range";
static const char idx_txt[] = "index '";
static const char out_of_range_txt[] = "' out of range";
}  // namespace

void ov::NodeValidationFailure::create(const char* file,
                                       int line,
                                       const char* check_string,
                                       const Node* node,
                                       const std::string& explanation) {
    throw ov::NodeValidationFailure(
        make_what(file, line, check_string, node_validation_failure_loc_string(node), explanation));
}

template <>
void ov::NodeValidationFailure::create(const char* file,
                                       int line,
                                       const char* check_string,
                                       std::pair<const Node*, const std::vector<PartialShape>*>&& ctx,
                                       const std::string& explanation) {
    throw ov::NodeValidationFailure(make_what(file,
                                              line,
                                              check_string,
                                              node_validation_failure_loc_string(ctx.first),
                                              op::validate::shape_infer_explanation_str(*ctx.second, explanation)));
}

atomic<size_t> ov::Node::m_next_instance_id(0);

ov::Node::Node() = default;

ov::Node::Node(const Node& node)
    : m_control_dependents(node.m_control_dependents),
      m_control_dependencies(node.m_control_dependencies),
      m_instance_id(m_next_instance_id.fetch_add(1)),
      m_friendly_name(node.m_friendly_name)
      // skip m_unique_name -- will be generated automatically
      ,
      m_inputs(node.m_inputs)  // will be modified in the body
      // skip m_outputs -- should be initialized outside
      ,
      m_rt_info(node.m_rt_info) {
    // cannot do it without copying node.m_inputs first due to too limiting const qualifiers
    for (auto& input : m_inputs) {
        input = descriptor::Input(this, input.get_index(), input.get_output());
        input.get_output().add_input(&input);
    }
}

ov::Node& ov::Node::operator=(const Node& node) {
    this->m_control_dependents = node.m_control_dependents;
    this->m_control_dependencies = node.m_control_dependencies;
    this->m_instance_id = m_next_instance_id.fetch_add(1);
    this->m_friendly_name = node.m_friendly_name;
    this->m_inputs = node.m_inputs;
    this->m_rt_info = node.m_rt_info;
    // cannot do it without copying node.m_inputs first due to too limiting const qualifiers
    for (auto& input : m_inputs) {
        input = descriptor::Input(this, input.get_index(), input.get_output());
        input.get_output().add_input(&input);
    }
    return *this;
}

void ov::Node::insert_info(std::shared_ptr<SharedRTInfo> info) {
    std::lock_guard<std::mutex> lock(m_insert_mutex);
    m_shared_rt_info.insert(std::move(info));
}

ov::Node::Node(size_t output_size) : Node() {
    set_output_size(output_size);
}

ov::Node::Node(const OutputVector& arguments, size_t output_size) : Node() {
    set_arguments(arguments);
    set_output_size(output_size);
}

ov::Node::~Node() {
    try {
        // raise a flag to reset nodes cache
        for_each(m_shared_rt_info.cbegin(), m_shared_rt_info.cend(), [](const std::shared_ptr<SharedRTInfo>& info) {
            info->set_use_topological_cache(false);
        });

        for (descriptor::Input& input : m_inputs) {
            if (input.has_output()) {
                // This test adds 1 to the actual count, so a count of 2 means this input is the only
                // reference to the node.
                if (input.get_output().get_node().use_count() == 2) {
                    // Don't want to trigger a deep recursive delete
                    NodeVector nodes{input.get_output().get_node()};
                    input.remove_output();
                    safe_delete(nodes, true);
                    return;
                }
                input.remove_output();
            }
        }
    } catch (...) {
    }
}

std::shared_ptr<ov::Node> ov::Node::copy_with_new_inputs(const OutputVector& inputs) const {
    return copy_with_new_inputs(inputs, get_control_dependencies());
}

ov::Output<const ov::Node> ov::Node::get_default_output() const {
    return output(get_default_output_index());
}

ov::Output<ov::Node> ov::Node::get_default_output() {
    return output(get_default_output_index());
}

size_t ov::Node::get_default_output_index() const {
    return 0;
}

size_t ov::Node::no_default_index() const {
    NODE_VALIDATION_CHECK(this, false, "Default output not supported");
}

std::shared_ptr<ov::Node> ov::Node::copy_with_new_inputs(
    const OutputVector& inputs,
    const std::vector<std::shared_ptr<Node>>& control_dependencies) const {
    shared_ptr<Node> clone = clone_with_new_inputs(inputs);
    for (auto& cdep : control_dependencies) {
        clone->add_control_dependency(cdep);
    }
    for (size_t i = 0; i < get_output_size(); i++) {
        clone->get_output_tensor(i).set_names(get_output_tensor(i).get_names());
        OPENVINO_SUPPRESS_DEPRECATED_START
        ov::descriptor::set_ov_tensor_legacy_name(clone->get_output_tensor(i),
                                                  ov::descriptor::get_ov_tensor_legacy_name(get_output_tensor(i)));
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
    return clone;
}

void ov::Node::safe_delete(NodeVector& nodes, bool recurse) {
    for (auto& input : m_inputs) {
        if (input.has_output()) {
            // This test adds 1 to the actual count, so a count of 2 means this input is the only
            // reference to the node.
            auto node = input.get_output().get_node();
            if (node.use_count() == 2) {
                // Move the node from the input to nodes so we don't trigger a deep recursive delete
                nodes.push_back(node);
            }
            input.remove_output();
        }
    }
    if (recurse) {
        while (nodes.size() > 0) {
            auto node = nodes.back();
            nodes.pop_back();
            node->safe_delete(nodes, false);
        }
    }
}

void ov::Node::set_arguments(const NodeVector& arguments) {
    OutputVector outputs;
    for (const auto& arg : arguments) {
        for (auto& output : arg->outputs()) {
            outputs.push_back(output);
        }
    }
    set_arguments(outputs);
}

void ov::Node::set_arguments(const OutputVector& arguments) {
    // Remove existing inputs of this node
    m_inputs.clear();

    // Add this node as a user of each argument.
    size_t i = 0;
    for (auto& output : arguments) {
        set_argument(i++, output);
    }

    // set_arguments doesn't use replace_output method, so we have to reset cache manually here
    for_each(this->m_shared_rt_info.cbegin(), this->m_shared_rt_info.cend(), [](std::shared_ptr<SharedRTInfo> info) {
        info->set_use_topological_cache(false);
    });
}

ov::descriptor::Input& ov::Node::get_input_descriptor(size_t position) {
    while (m_inputs.size() <= position) {
        m_inputs.emplace_back(this, m_inputs.size());
    }
    return m_inputs.at(position);
}

ov::descriptor::Output& ov::Node::get_output_descriptor(size_t position) {
    while (m_outputs.size() <= position) {
        size_t i = m_outputs.size();
        auto tensor_descriptor = make_shared<descriptor::Tensor>(element::dynamic, PartialShape::dynamic(), this, i);
        m_outputs.emplace_back(this, i, tensor_descriptor);
    }
    return m_outputs[position];
}

void ov::Node::set_argument(size_t position, const Output<Node>& argument) {
    auto output_node = argument.get_node();
    auto& output_descriptor = output_node->m_outputs.size() > argument.get_index()
                                  ? output_node->m_outputs.at(argument.get_index())
                                  : output_node->get_output_descriptor(argument.get_index());
    if (position < m_inputs.size()) {
        get_input_descriptor(position).replace_output(output_descriptor);
    } else {
        while (m_inputs.size() < position) {
            m_inputs.emplace_back(this, m_inputs.size());
        }
        m_inputs.emplace_back(this, position, output_descriptor);
    }
}

void ov::Node::constructor_validate_and_infer_types() {
    validate_and_infer_types();
}

void ov::Node::set_output_size(size_t n) {
    OPENVINO_ASSERT(n >= m_outputs.size(), "shrinking ", m_outputs.size(), " to ", n);
    for (size_t i = m_outputs.size(); i < n; ++i) {
        // create the descriptors
        get_output_descriptor(i);
    }
}

void ov::Node::invalidate_values() {
    for (const auto& output : outputs())
        output.get_tensor().invalidate_values();
}

void ov::Node::validate_and_infer_types() {}

void ov::Node::set_input_is_relevant_to_shape(size_t i, bool relevant) {
    OPENVINO_ASSERT(i < m_inputs.size(), idx_txt, i, out_of_range_txt);
    m_inputs[i].m_is_relevant_to_shape = relevant;
}

void ov::Node::set_input_is_relevant_to_value(size_t i, bool relevant) {
    OPENVINO_ASSERT(i < m_inputs.size(), idx_txt, i, out_of_range_txt);
    m_inputs[i].m_is_relevant_to_value = relevant;
}

void ov::Node::set_output_type(size_t i, const element::Type& element_type, const PartialShape& pshape) {
    ov::descriptor::set_tensor_type(get_output_descriptor(i).get_tensor(), element_type, pshape);
}

std::string ov::Node::description() const {
    return get_type_name();
}

const std::string& ov::Node::get_friendly_name() const {
    if (m_friendly_name.empty()) {
        return get_name();
    }
    return m_friendly_name;
}

const std::string& ov::Node::get_name() const {
    AtomicGuard lock(m_name_changing);
    if (m_unique_name.empty())
        m_unique_name = description() + "_" + to_string(m_instance_id);
    return m_unique_name;
}

void ov::Node::set_friendly_name(const string& name) {
    m_friendly_name = name;
}

ov::Node* ov::Node::get_input_node_ptr(size_t index) const {
    OPENVINO_ASSERT(index < m_inputs.size(), idx_txt, index, out_of_range_txt);
    return m_inputs[index].get_output().get_node().get();
}

std::shared_ptr<ov::Node> ov::Node::get_input_node_shared_ptr(size_t index) const {
    OPENVINO_ASSERT(index < m_inputs.size(), idx_txt, index, out_of_range_txt);
    return m_inputs[index].get_output().get_node();
}

ov::Output<ov::Node> ov::Node::get_input_source_output(size_t i) const {
    return input(i).get_source_output();
}

const std::vector<std::shared_ptr<ov::Node>>& ov::Node::get_control_dependencies() const {
    return m_control_dependencies;
}

const std::vector<ov::Node*>& ov::Node::get_control_dependents() const {
    return m_control_dependents;
}

void ov::Node::add_control_dependency(std::shared_ptr<Node> node) {
    if (find(m_control_dependencies.begin(), m_control_dependencies.end(), node) == m_control_dependencies.end()) {
        m_control_dependencies.push_back(node);
        if (find(node->m_control_dependents.begin(), node->m_control_dependents.end(), this) ==
            node->m_control_dependents.end()) {
            node->m_control_dependents.push_back(this);
        }
    }

    // control dependency may change the topological order so we have to reset cache
    // by setting a flag into shared node info.
    for_each(node->m_shared_rt_info.cbegin(), node->m_shared_rt_info.cend(), [](std::shared_ptr<SharedRTInfo> info) {
        info->set_use_topological_cache(false);
    });
}

void ov::Node::add_node_control_dependencies(const std::shared_ptr<const Node>& source_node) {
    for (auto& node : source_node->get_control_dependencies()) {
        add_control_dependency(node);
    }
}

void ov::Node::add_node_control_dependents(const std::shared_ptr<const Node>& source_node) {
    for (Node* node : source_node->get_control_dependents()) {
        node->add_control_dependency(shared_from_this());
    }
}

void ov::Node::transfer_control_dependents(std::shared_ptr<Node> replacement) {
    replacement->add_node_control_dependents(shared_from_this());
    clear_control_dependents();
}

void ov::Node::remove_control_dependency(std::shared_ptr<Node> node) {
    {
        auto it = find(m_control_dependencies.begin(), m_control_dependencies.end(), node);
        if (it != m_control_dependencies.end()) {
            m_control_dependencies.erase(it);
        }
    }
    {
        auto it = find(node->m_control_dependents.begin(), node->m_control_dependents.end(), this);
        if (it != node->m_control_dependents.end()) {
            node->m_control_dependents.erase(it);
        }
    }
}

void ov::Node::clear_control_dependencies() {
    for (auto& node : m_control_dependencies) {
        auto it = find(node->m_control_dependents.begin(), node->m_control_dependents.end(), this);
        if (it != node->m_control_dependents.end()) {
            node->m_control_dependents.erase(it);
        }
    }
    m_control_dependencies.clear();
}

void ov::Node::clear_control_dependents() {
    while (!m_control_dependents.empty()) {
        (*m_control_dependents.begin())->remove_control_dependency(shared_from_this());
    }
}

const ov::op::AutoBroadcastSpec& ov::Node::get_autob() const {
    static ov::op::AutoBroadcastSpec s_spec;
    return s_spec;
}

namespace ov {
ostream& operator<<(ostream& out, const Node& node) {
    return node.write_description(out, 1);
}
ostream& operator<<(ostream& out, const Node* node) {
    return node->write_description(out, 1);
}
}  // namespace ov

std::ostream& ov::Node::write_description(std::ostream& out, uint32_t depth) const {
    auto version = get_type_info().version_id;
    if (version)
        out << version << "::" << get_type_info().name << " " << get_friendly_name();
    else
        out << get_type_info().name << " " << get_friendly_name();

    if (depth > 0) {
        out << " (";
        string sep = "";
        for (const auto& arg : input_values()) {
            out << sep << arg;
            sep = ", ";
        }
        out << ") -> (";
        sep = "";
        for (size_t i = 0; i < get_output_size(); i++) {
            out << sep << get_output_element_type(i) << get_output_partial_shape(i);
            sep = ", ";
        }
        out << ")";
    }
    return out;
}

size_t ov::Node::get_output_size() const {
    return m_outputs.size();
}

const ov::element::Type& ov::Node::get_output_element_type(size_t i) const {
    OPENVINO_ASSERT(i < m_outputs.size(), idx_txt, i, out_of_range_txt);
    return m_outputs[i].get_element_type();
}

const ov::element::Type& ov::Node::get_element_type() const {
    if (get_output_size() != 1) {
        OPENVINO_THROW("get_element_type() must be called on a node with exactly one output.");
    }
    return get_output_element_type(0);
}

const ov::Shape& ov::Node::get_output_shape(size_t i) const {
    OPENVINO_ASSERT(i < m_outputs.size(), idx_txt, i, out_of_range_txt);
    return m_outputs[i].get_shape();
}

const ov::PartialShape& ov::Node::get_output_partial_shape(size_t i) const {
    OPENVINO_ASSERT(i < m_outputs.size(), idx_txt, i, out_of_range_txt);
    return m_outputs[i].get_partial_shape();
}

const ov::Shape& ov::Node::get_shape() const {
    NODE_VALIDATION_CHECK(this, get_output_size() == 1, "get_shape() must be called on a node with exactly one output");
    return get_output_shape(0);
}

std::set<ov::Input<ov::Node>> ov::Node::get_output_target_inputs(size_t i) const {
    std::set<Input<Node>> result;

    for (auto& input : m_outputs.at(i).get_inputs()) {
        result.emplace(input->get_raw_pointer_node(), input->get_index());
    }

    return result;
}

ov::descriptor::Tensor& ov::Node::get_output_tensor(size_t i) const {
    OPENVINO_ASSERT(i < m_outputs.size(), idx_txt, i, out_of_range_txt);
    return m_outputs[i].get_tensor();
}

ov::descriptor::Tensor& ov::Node::get_input_tensor(size_t i) const {
    OPENVINO_ASSERT(i < m_inputs.size(), idx_txt, i, out_of_range_txt);
    descriptor::Input input = m_inputs[i];
    return input.get_tensor();
}

size_t ov::Node::get_input_size() const {
    return m_inputs.size();
}

const ov::element::Type& ov::Node::get_input_element_type(size_t i) const {
    OPENVINO_ASSERT(i < m_inputs.size(), idx_txt, i, out_of_range_txt);
    return m_inputs[i].get_element_type();
}

const ov::Shape& ov::Node::get_input_shape(size_t i) const {
    OPENVINO_ASSERT(i < m_inputs.size(), idx_txt, i, out_of_range_txt);
    return m_inputs[i].get_shape();
}

const ov::PartialShape& ov::Node::get_input_partial_shape(size_t i) const {
    OPENVINO_ASSERT(i < m_inputs.size(), idx_txt, i, out_of_range_txt);
    return m_inputs[i].get_partial_shape();
}

bool ov::Node::has_same_type(std::shared_ptr<const Node> node) const {
    if (get_output_size() != node->get_output_size()) {
        return false;
    }
    for (size_t i = 0; i < get_output_size(); ++i) {
        if (get_output_element_type(i) != node->get_output_element_type(i) ||
            get_output_shape(i) != node->get_output_shape(i)) {
            return false;
        }
    }
    return true;
}

namespace ov {
bool is_used(Node* node);
}

ov::NodeVector ov::Node::get_users(bool check_is_used) const {
    NodeVector result;
    for (const auto& output : outputs()) {
        for (auto input : output.get_target_inputs()) {
            Node* input_node = input.get_node();
            if (!check_is_used || is_used(input_node)) {
                result.push_back(input_node->shared_from_this());
            }
        }
    }
    return result;
}

std::string ov::node_validation_failure_loc_string(const Node* node) {
    std::stringstream ss;
    ss << "While validating node '" << *node << "' with friendly_name '" << node->get_friendly_name() << '\'';
    return ss.str();
}

bool ov::Node::match_value(ov::pass::pattern::Matcher* matcher,
                           const Output<Node>& pattern_value,
                           const Output<Node>& graph_value) {
    if (pattern_value.get_index() != graph_value.get_index() ||
        (matcher->is_strict_mode() &&
         (!pattern_value.get_element_type().compatible(graph_value.get_element_type()) ||
          !pattern_value.get_partial_shape().compatible(graph_value.get_partial_shape())))) {
        return false;
    }
    return match_node(matcher, graph_value);
}

bool ov::Node::match_node(ov::pass::pattern::Matcher* matcher, const Output<Node>& graph_value) {
    matcher->add_node(graph_value);
    // Check if a type of a given node, which produces graph_value, matches the type of `this` node
    // or `this` node type is an ancestor of that node type. It is not the exact matching, types of
    // the nodes
    // may not match, but they are connected by the inheritance relation.
    // Not exact matching allows using base classes in the patterns and successfully matching such
    // patterns
    // with sub-graph of descent nodes types.
    if (graph_value.get_node_shared_ptr()->get_type_info().is_castable(get_type_info()) &&
        matcher->match_arguments(this, graph_value.get_node_shared_ptr())) {
        auto& pattern_map = matcher->get_pattern_value_map();
        pattern_map[shared_from_this()] = graph_value;
        return true;
    }
    return false;
}

// default implementation for the node to check if it contains partial shape
// we will override this method, for the Op's which depends on additional shape
// attribute to determine if node contains partial shape or not
bool ov::Node::is_dynamic() const {
    for (size_t i = 0; i < get_input_size(); i++) {
        if (get_input_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    return false;
}

ov::Input<ov::Node> ov::Node::input(size_t input_index) {
    if (input_index >= m_inputs.size()) {
        OPENVINO_THROW(node_idx_out_of_range_txt);
    }

    return {this, input_index};
}

ov::Output<ov::Node> ov::Node::input_value(size_t input_index) const {
    return input(input_index).get_source_output();
}

ov::Input<const ov::Node> ov::Node::input(size_t input_index) const {
    if (input_index >= m_inputs.size()) {
        OPENVINO_THROW(node_idx_out_of_range_txt);
    }

    return {this, input_index};
}

ov::Output<ov::Node> ov::Node::output(size_t output_index) {
    // All nodes will have at least 1 output
    if (output_index > 0 && output_index >= m_outputs.size()) {
        OPENVINO_THROW(node_idx_out_of_range_txt);
    }

    return Output<Node>(this, output_index);
}

ov::Output<const ov::Node> ov::Node::output(size_t output_index) const {
    // All nodes will have at least 1 output
    if (output_index > 0 && output_index >= m_outputs.size()) {
        OPENVINO_THROW(node_idx_out_of_range_txt);
    }

    return Output<const Node>(this, output_index);
}

vector<ov::Input<ov::Node>> ov::Node::inputs() {
    vector<Input<Node>> result;

    for (size_t i = 0; i < get_input_size(); i++) {
        result.emplace_back(this, i);
    }

    return result;
}

vector<ov::Output<ov::Node>> ov::Node::input_values() const {
    vector<Output<Node>> result;

    for (size_t i = 0; i < get_input_size(); i++) {
        result.emplace_back(input(i).get_source_output());
    }

    return result;
}

vector<ov::Input<const ov::Node>> ov::Node::inputs() const {
    vector<Input<const Node>> result;

    for (size_t i = 0; i < get_input_size(); i++) {
        result.emplace_back(this, i);
    }

    return result;
}

vector<ov::Output<ov::Node>> ov::Node::outputs() {
    vector<Output<Node>> result;

    for (size_t i = 0; i < get_output_size(); i++) {
        result.emplace_back(shared_from_this(), i);
    }

    return result;
}

vector<ov::Output<const ov::Node>> ov::Node::outputs() const {
    vector<Output<const Node>> result;

    for (size_t i = 0; i < get_output_size(); i++) {
        result.emplace_back(shared_from_this(), i);
    }

    return result;
}

bool ov::Node::has_evaluate() const {
    return false;
}

bool ov::Node::evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values) const {
    return false;
}

bool ov::Node::evaluate(ov::TensorVector& output_values,
                        const ov::TensorVector& input_values,
                        const ov::EvaluationContext& evaluationContext) const {
    // As nodes in most cases implements evaluate without context try call it unless override by child class
    return evaluate(output_values, input_values);
}

bool ov::Node::evaluate_lower(ov::TensorVector& output_values) const {
    const auto& inputs = input_values();
    const auto all_have_bounds = std::all_of(inputs.begin(), inputs.end(), [](const Output<Node>& output) {
        return output.get_tensor().has_and_set_bound();
    });
    return all_have_bounds && ov::default_lower_bound_evaluator(this, output_values);
}

bool ov::Node::evaluate_upper(ov::TensorVector& output_values) const {
    const auto& inputs = input_values();
    const auto all_have_bounds = std::all_of(inputs.begin(), inputs.end(), [](const Output<Node>& output) {
        return output.get_tensor().has_and_set_bound();
    });
    return all_have_bounds && ov::default_upper_bound_evaluator(this, output_values);
}

bool ov::Node::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return false;
}

bool ov::Node::can_constant_fold(const OutputVector& input_values) const {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Node::can_constant_fold");

    if (is_const_fold_disabled()) {
        return false;
    }

    // If all the inputs are constants, try to evaluate the outputs
    bool all_constants = std::all_of(input_values.begin(), input_values.end(), [](const Output<Node>& input) {
        return ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
    });

    return all_constants;
}

bool ov::Node::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::core, "Node::constant_fold");

    if (!Node::can_constant_fold(input_values)) {
        return false;
    }

    NodeVector nodes;
    TensorVector input_tensors;
    for (const auto& input : input_values) {
        nodes.push_back(input.get_node_shared_ptr());
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
        void* data = (void*)constant->get_data_ptr();
        auto tensor = ov::Tensor(input.get_element_type(), input.get_shape(), data);
        input_tensors.push_back(tensor);
    }

    TensorVector output_tensors;
    for (const auto& output : outputs()) {
        const auto& et = output.get_element_type();
        if (et != element::undefined && et.is_static()) {
            output_tensors.emplace_back(output);
        } else {
            output_tensors.emplace_back();
        }
    }

    if (evaluate(output_tensors, input_tensors)) {
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            output_values[i] = make_shared<ov::op::v0::Constant>(output_tensors[i]);
            ov::copy_runtime_info(nodes, output_values[i].get_node_shared_ptr());
        }
        return true;
    }
    return false;
}

bool ov::Node::is_const_fold_disabled() const {
    return ov::pass::constant_folding_is_disabled(this);
}

bool ov::Node::visit_attributes(AttributeVisitor&) {
    return true;
}

namespace ov {
void check_new_args_count(const Node* const node, const OutputVector& new_args) {
    NODE_VALIDATION_CHECK(node,
                          new_args.size() == node->input_values().size(),
                          "clone_with_new_inputs() expected ",
                          node->input_values().size(),
                          " argument",
                          (node->input_values().size() == 1 ? "" : "s"),
                          " but got ",
                          new_args.size());
}

AttributeAdapter<std::shared_ptr<Node>>::AttributeAdapter(std::shared_ptr<Node>& value) : m_ref(value) {}

bool AttributeAdapter<std::shared_ptr<Node>>::visit_attributes(AttributeVisitor& visitor) {
    auto original_id = visitor.get_registered_node_id(m_ref);
    auto id = original_id;
    visitor.on_attribute("ID", id);
    if (id != original_id) {
        m_ref = visitor.get_registered_node(id);
    }
    return true;
}

AttributeAdapter<NodeVector>::AttributeAdapter(NodeVector& ref) : m_ref(ref) {}

bool AttributeAdapter<NodeVector>::visit_attributes(AttributeVisitor& visitor) {
    size_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size()) {
        m_ref.resize(size);
    }
    ostringstream index;
    for (size_t i = 0; i < size; i++) {
        index.str("");
        index << i;
        string id;
        if (m_ref[i]) {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i]) {
            m_ref[i] = visitor.get_registered_node(std::move(id));
        }
    }
    return true;
}
}  // namespace ov
