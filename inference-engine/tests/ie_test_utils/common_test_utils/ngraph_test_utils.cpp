// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_test_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <ngraph/function.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>
namespace {
inline namespace tools {
bool isTypeRelaxed(const std::string &type) {
    return type.find_first_of("TypeRelaxed") == 0;
}

bool compareTypeInfo(const ngraph::DiscreteTypeInfo &info1, const ngraph::DiscreteTypeInfo &info2) {
    if (!isTypeRelaxed(info1.name) && !isTypeRelaxed(info2.name) &&
        (info1.version != info2.version)) {
        return false;
    }

    const std::string info1Name =
            isTypeRelaxed(info1.name) && (info1.parent != nullptr) ? info1.parent->name : info1.name;
    const std::string info2Name =
            isTypeRelaxed(info2.name) && (info2.parent != nullptr) ? info2.parent->name : info2.name;
    return info1Name == info2Name;
}

template<typename Node>
bool compare_rt_keys(const Node &node1, const Node &node2) {
    const auto &first_node_rt_info = node1->get_rt_info();
    const auto &second_node_rt_info = node2->get_rt_info();

    if (first_node_rt_info.empty() && second_node_rt_info.empty()) {
        return true;
    }

    if (first_node_rt_info.size() != second_node_rt_info.size()) {
        return false;
    }

    auto first_node_rt_info_it = first_node_rt_info.begin();
    auto second_node_rt_info_it = second_node_rt_info.begin();
    while (first_node_rt_info_it != first_node_rt_info.end()) {
        if (first_node_rt_info_it->first != second_node_rt_info_it->first) {
            return false;
        }
        ++first_node_rt_info_it;
        ++second_node_rt_info_it;
    }

    return true;
}

bool less_by_name(
        const std::shared_ptr<ngraph::op::v0::Result> &l,
        const std::shared_ptr<ngraph::op::v0::Result> &r) {
    return l->get_friendly_name() < r->get_friendly_name();
}

bool less_by_parent_name(
        const std::shared_ptr<ngraph::op::v0::Result> &l,
        const std::shared_ptr<ngraph::op::v0::Result> &r) {
    return l->get_input_node_shared_ptr(0)->get_friendly_name() < r->get_input_node_shared_ptr(0)->get_friendly_name();
}

std::string typeInfoToStr(const ngraph::Node::type_info_t &typeInfo) {
    return std::string(typeInfo.name) + "/" + to_str(typeInfo.version);
}


std::string tensor_names(const ngraph::descriptor::Tensor &t) {
    std::string n;
    const char *glue = "";
    for (const auto &name : t.get_names()) {
        n.append(glue).append(name);
        glue = ", ";
    }
    return "\"" + n + "\"";
}
}  // namespace tools

namespace subgraph {

namespace detail {

template<typename Ptr>
Ptr not_null(Ptr &&p) {
    if (!p) {
        IE_THROW() << "empty pointer";
    }
    return std::forward<Ptr>(p);
}

template<typename InOut1, typename InOut2>
bool equal_type_and_partial_shape(const InOut1 &lhs, const InOut2 &rhs) {
    return lhs.get_element_type() == rhs.get_element_type() &&
           lhs.get_partial_shape() == rhs.get_partial_shape();
}

class NodeAndInputDescription {
public:
    using SubGraphOp = ngraph::op::util::SubGraphOp;
    using InputDescripton = SubGraphOp::InputDescription;
    using InputNode = ngraph::Input<ngraph::Node>;
    using Parameter = ngraph::opset6::Parameter;

    explicit NodeAndInputDescription(
            const InputNode &input, const Parameter *parameter, const InputDescripton *description)
            : m_input(input), m_parameter(not_null(parameter)), m_description(not_null(description)) {}

    static bool equal_descriptions(const InputDescripton *lhs, const InputDescripton *rhs) {
        if (!lhs || !rhs || lhs->get_type_info() != rhs->get_type_info()) {
            return false;
        }

        if (lhs->get_type_info() == SubGraphOp::SliceInputDescription::type_info) {
            using InDesc = SubGraphOp::SliceInputDescription;
            const InDesc *l_input = static_cast<const InDesc *>(lhs);
            const InDesc *r_input = static_cast<const InDesc *>(rhs);
            return l_input->m_start == r_input->m_start && l_input->m_stride == r_input->m_stride &&
                   l_input->m_part_size == r_input->m_part_size &&
                   l_input->m_end == r_input->m_end && l_input->m_axis == r_input->m_axis;
        } else if (lhs->get_type_info() == SubGraphOp::MergedInputDescription::type_info) {
            return true;  // noting extra to check
        } else if (lhs->get_type_info() == SubGraphOp::InvariantInputDescription::type_info) {
            return true;  // noting extra to check
        }

        IE_THROW() << "Type is not supported: [" << lhs->get_type_info().name << "]";

        return false;
    }

    bool parameter_and_input_match(size_t num_iterations) const {
        if (const SubGraphOp::SliceInputDescription *slice_description =
                ngraph::as_type<const SubGraphOp::SliceInputDescription>(m_description)) {
            if (m_parameter->get_element_type() != m_input.get_element_type()) {
                return false;
            }
            const auto &param_partial_shape = m_parameter->get_partial_shape();
            const auto &input_partial_shape = m_input.get_partial_shape();
            if (param_partial_shape.is_dynamic() && input_partial_shape.is_dynamic()) {
                return true;
            }
            if (!param_partial_shape.is_static() || !input_partial_shape.is_static()) {
                return false;
            }
            const auto &param_shape = param_partial_shape.to_shape();
            const auto &input_shape = input_partial_shape.to_shape();
            if (param_shape.size() != input_shape.size()) {
                return false;
            }
            if (param_shape[slice_description->m_axis] != slice_description->m_part_size) {
                return false;
            }
            for (size_t i = 0; i != param_shape.size(); ++i) {
                const auto expected_axis_size =
                        i == slice_description->m_axis ? slice_description->m_part_size * num_iterations
                                                       : param_shape[i];
                if (input_shape[i] != expected_axis_size) {
                    return false;
                }
            }
            return true;
        } else if (
                m_description->get_type_info() == SubGraphOp::MergedInputDescription::type_info ||
                m_description->get_type_info() == SubGraphOp::InvariantInputDescription::type_info) {
            return equal_type_and_partial_shape(*m_parameter, m_input);
        }

        IE_THROW() << "Type is not supported: [" << m_description->get_type_info().name
                           << "]";

        return false;
    }

    static bool equal_parameters(const Parameter *lhs, const Parameter *rhs) {
        return lhs && rhs && equal_type_and_partial_shape(*lhs, *rhs);
    }

    friend bool operator==(const NodeAndInputDescription &lhs, const NodeAndInputDescription &rhs) {
        if (!equal_descriptions(lhs.m_description, rhs.m_description)) {
            return false;
        }
        return equal_parameters(lhs.m_parameter, rhs.m_parameter);
    }

private:
    const InputNode m_input;
    const Parameter *m_parameter;
    const InputDescripton *m_description;
};

class NodeAndOutputDescription {
public:
    using SubGraphOp = ngraph::op::util::SubGraphOp;
    using OutputDescription = SubGraphOp::OutputDescription;
    using OutputNode = ngraph::Output<ngraph::Node>;
    using Result = ngraph::opset6::Result;

    explicit NodeAndOutputDescription(
            const OutputNode &output, const Result *result, const OutputDescription *description)
            : m_output(output), m_result(not_null(result)), m_description(not_null(description)) {}

    static bool equal_descriptions(const OutputDescription *lhs, const OutputDescription *rhs) {
        if (!lhs || !rhs || lhs->get_type_info() != rhs->get_type_info()) {
            return false;
        }

        if (lhs->get_type_info() == SubGraphOp::ConcatOutputDescription::type_info) {
            using OutDesc = SubGraphOp::ConcatOutputDescription;
            const OutDesc *l_output = static_cast<const OutDesc *>(lhs);
            const OutDesc *r_output = static_cast<const OutDesc *>(rhs);
            return l_output->m_start == r_output->m_start &&
                   l_output->m_stride == r_output->m_stride &&
                   l_output->m_part_size == r_output->m_part_size &&
                   l_output->m_end == r_output->m_end && l_output->m_axis == r_output->m_axis;
        } else if (lhs->get_type_info() == SubGraphOp::BodyOutputDescription::type_info) {
            using OutDesc = SubGraphOp::BodyOutputDescription;
            const OutDesc *l_output = static_cast<const OutDesc *>(lhs);
            const OutDesc *r_output = static_cast<const OutDesc *>(rhs);
            return l_output->m_iteration == r_output->m_iteration;
        }

        IE_THROW() << "Type is not supported: [" << lhs->get_type_info().name << "]";

        return false;
    }

    bool result_and_output_match(size_t num_iterations) const {
        if (const auto concat_desciption =
                ngraph::as_type<const SubGraphOp::ConcatOutputDescription>(m_description)) {
            if (m_result->output(0).get_element_type() != m_output.get_element_type()) {
                return false;
            }

            const auto &output_partial_shape = m_output.get_partial_shape();
            const auto &result_partial_shape = m_result->output(0).get_partial_shape();
            if (result_partial_shape.is_dynamic() && output_partial_shape.is_dynamic()) {
                return true;
            }
            if (!result_partial_shape.is_static() || !output_partial_shape.is_static()) {
                return false;
            }
            const auto &output_shape = output_partial_shape.to_shape();
            const auto &result_shape = result_partial_shape.to_shape();
            if (result_shape.size() != output_shape.size()) {
                return false;
            }
            for (size_t i = 0; i != result_shape.size(); ++i) {
                const auto axis_multiplier = i == concat_desciption->m_axis ? num_iterations : 1;
                if (result_shape[i] * axis_multiplier != output_shape[i]) {
                    return false;
                }
            }
            return true;
        } else if (m_description->get_type_info() == SubGraphOp::BodyOutputDescription::type_info) {
            return equal_type_and_partial_shape(m_result->output(0), m_output);
        }

        IE_THROW() << "Type is not supported: [" << m_description->get_type_info().name
                           << "]";

        return false;
    }

    static bool equal_results(const Result *lhs, const Result *rhs) {
        return lhs && rhs && equal_type_and_partial_shape(lhs->output(0), rhs->output(0));
    }

    friend bool operator==(
            const NodeAndOutputDescription &lhs, const NodeAndOutputDescription &rhs) {
        if (!equal_descriptions(lhs.m_description, rhs.m_description)) {
            return false;
        }
        return equal_results(lhs.m_result, rhs.m_result);
    }

private:
    const OutputNode m_output;
    const Result *m_result;
    const OutputDescription *m_description;
};

class BackEdge {
public:
    using Parameter = ngraph::opset6::Parameter;
    using Result = ngraph::opset6::Result;
    using Id = uint64_t;

    explicit BackEdge(const Parameter *parameter, const Result *result)
            : m_parameter(not_null(parameter)), m_result(not_null(result)) {}

    bool result_and_parameter_match() const {
        return equal_type_and_partial_shape(m_result->output(0), *m_parameter);
    }

    friend bool operator==(const BackEdge &lhs, const BackEdge &rhs) {
        return equal_type_and_partial_shape(*lhs.m_parameter, *rhs.m_parameter) &&
               equal_type_and_partial_shape(lhs.m_result->output(0), rhs.m_result->output(0));
    }

private:
    const Parameter *m_parameter;
    const Result *m_result;
};

std::vector<NodeAndInputDescription> extract_inputs(ngraph::op::util::SubGraphOp *sub) {
    std::vector<NodeAndInputDescription> nodes;
    const auto &fn_body = sub->get_function();
    const auto &fn_parameters = fn_body->get_parameters();

    for (const auto &in_desc : sub->get_input_descriptions()) {
        const auto parameter = fn_parameters.at(in_desc->m_body_parameter_index).get();
        const auto input = sub->input(in_desc->m_input_index);
        nodes.push_back(NodeAndInputDescription{input, parameter, in_desc.get()});
    }
    return nodes;
}

std::vector<NodeAndOutputDescription> extract_outputs(ngraph::op::util::SubGraphOp *sub) {
    std::vector<NodeAndOutputDescription> nodes;
    const auto &fn_body = sub->get_function();
    const auto &fs_results = fn_body->get_results();

    for (const auto &out_desc : sub->get_output_descriptions()) {
        const auto result = fs_results.at(out_desc->m_body_value_index).get();
        const auto output = sub->output(out_desc->m_output_index);
        nodes.push_back(NodeAndOutputDescription{output, result, out_desc.get()});
    }
    return nodes;
}

std::vector<BackEdge> extract_backedges(ngraph::op::util::SubGraphOp *sub) {
    using MergedInputDescription = ngraph::op::util::SubGraphOp::MergedInputDescription;
    std::vector<BackEdge> edges;
    const auto &fn_body = sub->get_function();

    const auto &fs_parameters = fn_body->get_parameters();
    const auto &fs_results = fn_body->get_results();

    for (const auto &in_desc : sub->get_input_descriptions()) {
        if (const auto &merged_in_desc =
                ngraph::as_type_ptr<const MergedInputDescription>(in_desc)) {
            const auto parameter = fs_parameters.at(merged_in_desc->m_body_parameter_index);
            const auto result = fs_results.at(merged_in_desc->m_body_value_index);
            edges.push_back(BackEdge{parameter.get(), result.get()});
        }
    }
    return edges;
}

struct NotValidInputOrOutput {
    NotValidInputOrOutput(int64_t num_iterations) : m_num_iterations(num_iterations) {}

    bool operator()(const NodeAndOutputDescription &d) const {
        return !d.result_and_output_match(m_num_iterations);
    }

    bool operator()(const NodeAndInputDescription &d) const {
        return !d.parameter_and_input_match(m_num_iterations);
    }

    int64_t m_num_iterations;
};

bool not_valid_back_edge(const BackEdge &be) {
    return !be.result_and_parameter_match();
}

bool equal_body_ports(ngraph::opset6::Loop *lhs, ngraph::opset6::Loop *rhs) {
    if (!lhs || !rhs) {
        return false;
    }
    const auto &lhs_fn_body = lhs->get_function();
    const auto &rhs_fn_body = rhs->get_function();

    const auto &lhs_sbp = lhs->get_special_body_ports();
    const auto &rhs_sbp = rhs->get_special_body_ports();

    constexpr int64_t port_not_provided = -1;

    const bool input_provided = lhs_sbp.current_iteration_input_idx != port_not_provided ||
                                rhs_sbp.current_iteration_input_idx != port_not_provided;

    if (input_provided) {
        const auto &lhs_parameter =
                lhs_fn_body->get_parameters().at(lhs_sbp.current_iteration_input_idx);
        const auto &rhs_parameter =
                rhs_fn_body->get_parameters().at(rhs_sbp.current_iteration_input_idx);
        if (!NodeAndInputDescription::equal_parameters(lhs_parameter.get(), rhs_parameter.get())) {
            return false;
        }
    }

    const auto &lhs_result = lhs_fn_body->get_results().at(lhs_sbp.body_condition_output_idx);
    const auto &rhs_result = rhs_fn_body->get_results().at(rhs_sbp.body_condition_output_idx);

    return NodeAndOutputDescription::equal_results(lhs_result.get(), rhs_result.get());
}

class CompareSubGraphs {
public:
    using Result = Comparator::Result;
    using SubGraphOp = ngraph::op::util::SubGraphOp;

    Result compare(SubGraphOp *sub_lhs, SubGraphOp *sub_rhs) {
        const auto lhs_it_no = get_num_iterations(sub_lhs);
        const auto rhs_it_no = get_num_iterations(sub_rhs);
        if (lhs_it_no != rhs_it_no) {
            return Result::error("different number of iterations");
        }

        not_valid_input_output = lhs_it_no;

        const auto result_for_inputs = compare_inputs(sub_lhs, sub_rhs);
        if (!result_for_inputs.valid) {
            return result_for_inputs;
        }

        const auto result_for_outputs = compare_outputs(sub_lhs, sub_rhs);
        if (!result_for_outputs.valid) {
            return result_for_outputs;
        }

        return compare_backedges(sub_lhs, sub_rhs);
    }

private:
    Result compare_inputs(SubGraphOp *sub_lhs, SubGraphOp *sub_rhs) const {
        const auto &lhs_sub_inputs = extract_inputs(sub_lhs);
        const auto &rhs_sub_inputs = extract_inputs(sub_rhs);

        if (lhs_sub_inputs.empty() || rhs_sub_inputs.empty()) {
            return Result::error("no input in subgraph");
        }

        if (std::any_of(begin(lhs_sub_inputs), end(lhs_sub_inputs), not_valid_input_output)) {
            return Result::error("inputs and parameters mismatch");
        }
        if (std::any_of(begin(rhs_sub_inputs), end(rhs_sub_inputs), not_valid_input_output)) {
            return Result::error("inputs and parameters mismatch");
        }

        if (lhs_sub_inputs.size() != rhs_sub_inputs.size() ||
            !std::is_permutation(
                    begin(lhs_sub_inputs), end(lhs_sub_inputs), begin(rhs_sub_inputs))) {
            return Result::error("different SubGraph InputDescription");
        }
        return Result::ok();
    }

    Result compare_outputs(SubGraphOp *sub_lhs, SubGraphOp *sub_rhs) const {
        const auto &lhs_sub_outputs = extract_outputs(sub_lhs);
        const auto &rhs_sub_outputs = extract_outputs(sub_rhs);

        if (lhs_sub_outputs.empty() || rhs_sub_outputs.empty()) {
            return Result::error("no output in subgraph");
        }

        if (std::any_of(begin(lhs_sub_outputs), end(lhs_sub_outputs), not_valid_input_output)) {
            return Result::error("outputs and results mismatch");
        }
        if (std::any_of(begin(rhs_sub_outputs), end(rhs_sub_outputs), not_valid_input_output)) {
            return Result::error("outputs and results mismatch");
        }

        if (lhs_sub_outputs.size() != rhs_sub_outputs.size() ||
            !std::is_permutation(
                    begin(lhs_sub_outputs), end(lhs_sub_outputs), begin(rhs_sub_outputs))) {
            return Result::error("different SubGraph OutputDescription");
        }
        return Result::ok();
    }

    Result compare_backedges(SubGraphOp *sub_lhs, SubGraphOp *sub_rhs) const {
        const auto lhs_back_edges = extract_backedges(sub_lhs);
        const auto rhs_back_edges = extract_backedges(sub_rhs);

        if (std::any_of(begin(lhs_back_edges), end(lhs_back_edges), not_valid_back_edge)) {
            return Result::error("back edges mismatch");
        }
        if (std::any_of(begin(rhs_back_edges), end(rhs_back_edges), not_valid_back_edge)) {
            return Result::error("back edges mismatch");
        }

        if (lhs_back_edges.size() != rhs_back_edges.size() ||
            !std::is_permutation(
                    begin(lhs_back_edges), end(lhs_back_edges), begin(rhs_back_edges))) {
            return Result::error("different SubGraph BackEdges");
        }
        if (auto loop_lhs = ngraph::as_type<ngraph::opset6::Loop>(sub_lhs)) {
            auto loop_rhs = ngraph::as_type<ngraph::opset6::Loop>(sub_rhs);
            if (!equal_body_ports(loop_lhs, loop_rhs)) {
                return Result::error("different Special Body Ports");
            }
        }
        return Result::ok();
    }

    static int64_t get_num_iterations(ngraph::op::util::SubGraphOp *sub) {
        using namespace ngraph::opset6;
        if (const auto ti = dynamic_cast<const TensorIterator *>(sub)) {
            return ti->get_num_iterations();
        }
        if (const auto l = dynamic_cast<const Loop *>(sub)) {
            return l->get_num_iterations();
        }

        return -1;
    }

    NotValidInputOrOutput not_valid_input_output{-1};
};

}  // namespace detail

Comparator::Result compare_io(
        ngraph::op::util::SubGraphOp *sub_lhs, ngraph::op::util::SubGraphOp *sub_rhs) {
    return detail::CompareSubGraphs{}.compare(sub_lhs, sub_rhs);
}
}  // namespace subgraph
}  // namespace
Comparator::Result Comparator::compare(
    const std::shared_ptr<ngraph::Function>& f1, const std::shared_ptr<ngraph::Function>& f2) {
    /*
     * This function compares two nGraph functions and requires them to have exactly one output
     * + Check nodes types
     * + Check number of inputs
     * + Check shapes
     * + Check parent ports
     * + Check node attributes by Visitor API
     */

    auto f1_results = f1->get_results();
    auto f2_results = f2->get_results();

    auto cmp = less_by_name;
    // In case if Result source output has more than one name so the Result may have any of this names as a friendly name
    // And in case of multiple names we sort Result operation using their parent node names
    if (std::any_of(f1_results.begin(), f1_results.end(), [](const std::shared_ptr<ngraph::Node> & node) {
        const auto & t = node->input_value(0).get_tensor_ptr();
        return t->get_names().size() > 1;
    }) || std::any_of(f2_results.begin(), f2_results.end(), [](const std::shared_ptr<ngraph::Node> & node) {
        const auto & t = node->input_value(0).get_tensor_ptr();
        return t->get_names().size() > 1;
    })) {
        cmp = less_by_parent_name;
    }

    std::sort(f1_results.begin(), f1_results.end(), cmp);
    std::sort(f2_results.begin(), f2_results.end(), cmp);

    if (f1_results.size() != f2_results.size()) {
        return Result::error(
            "Number of results is different: " + to_str(f1_results.size()) + " and " +
            to_str(f2_results.size()));
    }

    const auto& f1_sinks = f1->get_sinks();
    const auto& f2_sinks = f2->get_sinks();
    if (f1_sinks.size() != f2_sinks.size()) {
        return Result::error(
            "Number of sinks is different: " + to_str(f1_sinks.size()) + " and " +
            to_str(f2_sinks.size()));
    }

    for (size_t i = 0; i < f1_results.size(); ++i) {
        if (should_compare(CmpValues::NAMES)) {
            if (name(f1_results[i]->get_input_node_shared_ptr(0)) !=
                name(f2_results[i]->get_input_node_shared_ptr(0))) {
                return Result::error(
                    "Different output names: " + name(f1_results[i]->get_input_node_shared_ptr(0)) +
                    " and " + name(f2_results[i]->get_input_node_shared_ptr(0)));
            }
        }
        q.push({f1_results[i].get(), f2_results[i].get()});
        used.insert(f1_results[i].get());
    }

    std::stringstream errors;

    while (!q.empty()) {
        ngraph::Node* const node1 = q.front().first;
        ngraph::Node* const node2 = q.front().second;
        q.pop();

        const auto result = compare(node1, node2, errors);
        if (!result.valid) {
            return result;
        }

        add_nodes_inputs_to_queue(node1, node2);
    }
    const auto msg = errors.str();
    return msg.empty() ? Result::ok() : Result::error(msg);
}

Comparator::Result Comparator::compare(
    ngraph::Node* node1, ngraph::Node* node2, std::ostream& err_log) {
    auto type_info1 = node1->get_type_info();
    auto type_info2 = node2->get_type_info();

    if (!compareTypeInfo(type_info1, type_info2)) {
        return Result::error(typeInfoToStr(type_info1) + " != " + typeInfoToStr(type_info2));
    }

    auto subgraph1 = dynamic_cast<ngraph::op::util::SubGraphOp*>(node1);
    auto subgraph2 = dynamic_cast<ngraph::op::util::SubGraphOp*>(node2);

    const bool subgraph_nodes = subgraph1 && subgraph2;

    if (subgraph_nodes) {
        const auto result = subgraph::compare_io(subgraph1, subgraph2);
        if (!result.valid) {
            return result;
        }
    }

    const auto& dependencies_1 = node1->get_control_dependencies();
    const auto& dependencies_2 = node2->get_control_dependencies();

    if (dependencies_1.size() != dependencies_2.size()) {
        return Result::error(
            "Number of dependencies is different: " + to_str(dependencies_1.size()) + " for " +
            name(node1) + " and " + to_str(dependencies_2.size()) + " for " + name(node2));
    }

    if (node1->inputs().size() != node2->inputs().size()) {
        return Result::error(
            "Number of inputs is different: " + to_str(node1->inputs().size()) + " for " +
            name(node1) + " and " + to_str(node2->inputs().size()) + " for " + name(node2));
    }

    if (node1->outputs().size() != node2->outputs().size()) {
        return Result::error(
            "Number of outputs is different: " + to_str(node1->inputs().size()) + " for " +
            name(node1) + " and " + to_str(node2->inputs().size()) + " for " + name(node2));
    }

    if (!subgraph_nodes) {
        compare_inputs(node1, node2, err_log);
        compare_outputs(node1, node2, err_log);
    }

    if (should_compare(CmpValues::ATTRIBUTES)) {
        const auto result = attributes::compare(node1, node2, m_comparition_flags);
        if (!result.valid) {
            return result;
        }
    }

    return Result::ok("Check if any minor error was log in to err_log");
}


void Comparator::compare_inputs(ngraph::Node* node1, ngraph::Node* node2, std::ostream& err_log) {
    for (size_t i = 0; i < node1->inputs().size(); ++i) {
        if (should_compare(CmpValues::CONST_VALUES)) {
            using Constant = ngraph::opset1::Constant;
            const auto equal_value =
                ::attributes::detail::equal::Equal<std::shared_ptr<Constant>>::equal_value;

            auto const1 = ngraph::as_type_ptr<Constant>(node1->get_input_node_shared_ptr(i));
            auto const2 = ngraph::as_type_ptr<Constant>(node2->get_input_node_shared_ptr(i));
            if (const1 && const2 && !equal_value(const1, const2)) {
                err_log << "Different Constant values detected\n"
                        << node1->description() << " Input(" << i << ") and "
                        << node2->description() << " Input(" << i << ")" << std::endl;
            }
        }

        if (should_compare(CmpValues::PRECISIONS)) {
            if (node1->input(i).get_element_type() != node2->input(i).get_element_type()) {
                err_log << "Different element type detected\n"
                        << name(node1) << " Input(" << i << ") "
                        << node1->input(i).get_element_type() << " and " << name(node2) << " Input("
                        << i << ") " << node2->input(i).get_element_type() << std::endl;
            }
        }

        if (!node1->input(i).get_partial_shape().same_scheme(node2->input(i).get_partial_shape())) {
            err_log << "Different shape detected\n"
                    << name(node1) << " Input(" << i << ") " << node1->input(i).get_partial_shape()
                    << " and " << name(node2) << " Input(" << i << ") "
                    << node2->input(i).get_partial_shape() << std::endl;
        }

        if (node1->get_input_source_output(i).get_index() !=
            node2->get_input_source_output(i).get_index()) {
            auto idx1 = node1->get_input_source_output(i).get_index();
            auto idx2 = node2->get_input_source_output(i).get_index();
            err_log << "Different ports detected\n"
                    << name(node1) << " Input(" << i << ") connected to parent port " << idx1
                    << " and " << name(node2) << " Input(" << i << ") connected to parent port "
                    << idx2 << std::endl;
        }

        if (should_compare(CmpValues::RUNTIME_KEYS) && !compare_rt_keys(node1, node2)) {
            err_log << "Different runtime info detected\n"
                    << name(node1) << " and " << name(node2) << " not equal runtime info."
                    << std::endl;
        }
    }
}

void Comparator::compare_outputs(ngraph::Node* node1, ngraph::Node* node2, std::ostream& err_log) {
    for (int i = 0; i < node1->outputs().size(); ++i) {
        const auto& tensor1 = node1->output(i).get_tensor();
        const auto& tensor2 = node2->output(i).get_tensor();

        if (tensor1.get_names() != tensor2.get_names()) {
            err_log << "Output tensors names " << tensor_names(tensor1) << " and "
                    << tensor_names(tensor2)
                    << " are different for nodes: " << node1->get_friendly_name() << " and "
                    << node2->get_friendly_name() << std::endl;
        }

        if (!node1->output(i).get_partial_shape().same_scheme(
                node2->output(i).get_partial_shape())) {
            err_log << "Different shape detected\n"
                    << name(node1) << " Output(" << i << ") "
                    << node1->output(i).get_partial_shape() << " and " << name(node2) << " Output("
                    << i << ") " << node2->output(i).get_partial_shape() << std::endl;
        }
    }
}

void Comparator::add_nodes_inputs_to_queue(ngraph::Node* node1, ngraph::Node* node2) {
    for (int i = 0; i < node1->inputs().size(); ++i) {
        if (!used.count(node1->input_value(i).get_node())) {
            q.push({node1->input_value(i).get_node(), node2->input_value(i).get_node()});
            used.insert(node1->input_value(i).get_node());
        }
    }
}

FunctionsComparator::Result FunctionsComparator::compare(
    const std::shared_ptr<ngraph::Function>& f1,
    const std::shared_ptr<ngraph::Function>& f2) const {
    return Comparator(m_comparition_flags).compare(f1, f2);
}

void check_rt_info(const std::shared_ptr<ngraph::Function>& f) {
    static const std::vector<std::string> attrs_to_check{"Variant::RuntimeAttribute::FusedNames"};

    std::ostringstream err_log;
    for (auto& op : f->get_ops()) {
        if (ngraph::op::is_constant(op)) continue;

        const auto& rt_info = op->get_rt_info();
        for (const auto& attr_name : attrs_to_check) {
            if (!rt_info.count(attr_name)) {
                err_log << "Node: " << op->get_friendly_name() << " has no attribute: " << attr_name
                        << std::endl;
            }
        }
    }

    auto err_msg = err_log.str();
    if (!err_msg.empty()) {
        throw ngraph::ngraph_error(err_msg);
    }
}

NGRAPH_RTTI_DEFINITION(TestOpMultiOut, "TestOp", 0);

namespace attributes {
namespace detail {
void ReadAndStoreAttributes::on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) {
    if (auto inputs =
            ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpInputDescription>>(&adapter)) {
        insert(name, inputs->get());
    } else if (
            auto outputs =
                    ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpOutputDescription>>(&adapter)) {
        insert(name, outputs->get());
    } else if (ngraph::is_type<ngraph::AttributeAdapter<SpecialBodyPorts>>(&adapter)) {
        // drop comparison, no more info than port indexes which will be check in
        // subgraph::compare_io
    } else if (
            auto a = ngraph::as_type<
                    ngraph::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
                    &adapter)) {
        const auto beg = static_cast<unsigned char *>(a->get()->get_ptr());
        const auto end = beg + a->get()->size();
        insert(name, storage::MemoryChunk{storage::MemoryChunk::Data(beg, end)});
    } else if (auto framework_node_attr = ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::FrameworkNodeAttrs>>(&adapter)) {
        insert(name, framework_node_attr->get());
    } else {
        m_read_result += "store   attr [ ERR ]: " + name +
                         " [drop `void` comparison which is '" + adapter.get_type_info().name +
                         "']";
    }
}
template <typename AttrValue>
void ReadAndCompareAttributes::verify(const std::string &name, const AttrValue &attr_value) {
    if (should_return()) {
        return;
    }
    m_visited_attributes.insert(name);
    const auto ref_value = m_attr_ref.get<AttrValue>(name);
    if (!ref_value) {
        m_cmp_result += "missing attribute name: '" + name + "'";
        return;
    }

    if (!equal::Equal<AttrValue>::equal_value(*ref_value, attr_value)) {
        m_cmp_result += "mismatch in value: '" + name +
                        "' : " + str::Get<AttrValue>::value(*ref_value) + " vs " +
                        str::Get<AttrValue>::value(attr_value);
    }
}

void ReadAndCompareAttributes::verify_mem_buf(const std::string &name,
                                              const std::shared_ptr<ngraph::runtime::AlignedBuffer> &buffer) {
    if (should_return()) {
        return;
    }
    m_visited_attributes.insert(name);
    const auto ref_value = m_attr_ref.get<storage::MemoryChunk>(name);
    if (!ref_value) {
        m_cmp_result += "missing attribute name: '" + name + "'";
        return;
    }

    if (buffer->size() != ref_value->size() ||
        std::memcmp(ref_value->data(), buffer->get_ptr(), ref_value->size()) != 0) {
        m_cmp_result += "mismatch in value: '" + name + "' : look in to the mem buffer";
        return;
    }
}

void ReadAndCompareAttributes::verify_function(const std::string &name, FunctionAccessor &adapter) {
    if (should_return()) {
        return;
    }
    m_visited_attributes.insert(name);
    const auto ref_value = m_attr_ref.get<std::shared_ptr<ngraph::Function>>(name);
    if (!ref_value) {
        m_cmp_result += "missing attribute name: '" + name + "'";
        return;
    }
    Comparator c(m_check_flags);
    const auto result = c.compare(*ref_value, adapter.get());
    if (!result.valid) {
        m_cmp_result += result.message;
    }
}

void ReadAndCompareAttributes::verify_others(const std::string &name, ngraph::ValueAccessor<void> &adapter) {
    if (auto inputs =
            ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpInputDescription>>(&adapter)) {
        verify(name, inputs->get());
    } else if (
            auto outputs =
                    ngraph::as_type<ngraph::AttributeAdapter<SubGraphOpOutputDescription>>(&adapter)) {
        verify(name, outputs->get());
    } else if (ngraph::is_type<ngraph::AttributeAdapter<SpecialBodyPorts>>(&adapter)) {
        // drop comparison, no more info than port indexes which will be check in
        // subgraph::compare_io
    } else if (
            auto a = ngraph::as_type<
                    ngraph::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(
                    &adapter)) {
        verify_mem_buf(name, a->get());
    } else if (auto attrs = ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::FrameworkNodeAttrs>>(&adapter)) {
        verify(name, attrs->get());
    } else {
        m_cmp_result += "compare attr [ ERR ]: " + name +
                        " [drop `void` comparison which is '" + adapter.get_type_info().name +
                        "']";
    }
}

}  // namespace detail

Comparator::Result compare(
        ngraph::Node* node1, ngraph::Node* node2, Comparator::CmpValues comparition_flags) {
    detail::CompareNodesAttributes compare_nodes_attr(comparition_flags);
    node1->visit_attributes(compare_nodes_attr.get_ref_reader());
    node2->visit_attributes(compare_nodes_attr.get_cmp_reader());
    if (!compare_nodes_attr.equal()) {
        return Comparator::Result::error(
                "Comparison of attributes failed for nodes " + name(node1) + ", " + name(node2) +
                " [cmp status: " + to_str(compare_nodes_attr) + "]");
    }
    return Comparator::Result::ok(to_str(compare_nodes_attr));
}

}  // namespace attributes