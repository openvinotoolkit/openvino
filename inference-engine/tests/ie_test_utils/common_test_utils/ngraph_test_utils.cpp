// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_test_utils.hpp"

#include <cassert>
#include <memory>
#include <queue>
#include <string>

#include <ngraph/function.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/visualize_tree.hpp>

namespace {
bool isTypeRelaxed(const std::string& type) {
    return type.find_first_of("TypeRelaxed") == 0;
}

bool compareTypeInfo(const ngraph::DiscreteTypeInfo& info1, const ngraph::DiscreteTypeInfo& info2) {
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

template <typename Node>
bool compare_rt_keys(const Node& node1, const Node& node2) {
    const auto& first_node_rt_info = node1->get_rt_info();
    const auto& second_node_rt_info = node2->get_rt_info();

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
    const std::shared_ptr<ngraph::op::v0::Result>& l,
    const std::shared_ptr<ngraph::op::v0::Result>& r) {
    return l->get_friendly_name() < r->get_friendly_name();
}

template <typename T>
std::string to_str(const T& v) {
    return std::to_string(v);
}

std::pair<bool, std::string> error(std::string s) {
    return {false, std::move(s)};
}

std::string typeInfoToStr(const ngraph::Node::type_info_t& typeInfo) {
    return std::string(typeInfo.name) + "/" + to_str(typeInfo.version);
}

template <typename Node>
std::string name(const Node& n) {
    return n->get_friendly_name();
}

template <typename Constant>
bool equal(const Constant& c1, const Constant& c2) {
    const auto equal_float_str = [](const std::string& s1, const std::string s2) {
        return std::abs(std::stof(s1) - std::stof(s2)) < 0.001;
    };
    const auto& c1v = c1.get_value_strings();
    const auto& c2v = c2.get_value_strings();

    return c1v.size() == c2v.size() &&
           std::equal(begin(c1v), end(c1v), begin(c2v), equal_float_str);
}

}  // namespace

std::pair<bool, std::string> compare_functions(
    const std::shared_ptr<ngraph::Function>& f1,
    const std::shared_ptr<ngraph::Function>& f2,
    const bool compareConstValues,
    const bool compareNames,
    const bool compareRuntimeKeys,
    const bool comparePrecisions) {
    /*
     * This function compares two nGraph functions and requires them to have exactly one output
     * + Check nodes types
     * + Check number of inputs
     * + Check shapes
     * + Check parent ports
     * - Do not check nodes attributes (requires visitor mechanism to be completed)
     */

    auto f1_results = f1->get_results();
    auto f2_results = f2->get_results();

    std::sort(f1_results.begin(), f1_results.end(), less_by_name);
    std::sort(f2_results.begin(), f2_results.end(), less_by_name);

    if (f1_results.size() != f2_results.size()) {
        return error(
            "Number of results is different: " + to_str(f1_results.size()) + " and " + to_str(f2_results.size()));
    }

    const auto& f1_sinks = f1->get_sinks();
    const auto& f2_sinks = f2->get_sinks();
    if (f1_sinks.size() != f2_sinks.size()) {
        return error(
            "Number of sinks is different: " + to_str(f1_sinks.size()) + " and " + to_str(f2_sinks.size()));
    }

    std::ostringstream err_log;

    using ComparedNodes = std::pair<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>;
    std::queue<ComparedNodes> q;

    for (size_t i = 0; i < f1_results.size(); ++i) {
        if (compareNames) {
            if (name(f1_results[i]->get_input_node_shared_ptr(0)) !=
                name(f2_results[i]->get_input_node_shared_ptr(0))) {
                return error(
                    "Different output names: " + name(f1_results[i]->get_input_node_shared_ptr(0)) +
                    " and " + name(f2_results[i]->get_input_node_shared_ptr(0)));
            }
        }
        q.push({f1_results[i], f2_results[i]});
    }

    while (!q.empty()) {
        auto node1 = q.front().first;
        auto node2 = q.front().second;
        q.pop();

        auto type_info1 = node1->get_type_info();
        auto type_info2 = node2->get_type_info();

        if (!compareTypeInfo(type_info1, type_info2)) {
            return error(typeInfoToStr(type_info1) + " != " + typeInfoToStr(type_info2));
        }

        auto subgraph1 = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(node1);
        auto subgraph2 = std::dynamic_pointer_cast<ngraph::op::util::SubGraphOp>(node2);

        if (subgraph1 && subgraph2) {
            auto res = compare_functions(subgraph1->get_function(), subgraph2->get_function(),
                    compareConstValues, compareNames, compareRuntimeKeys, comparePrecisions);
            if (!res.first) {
                return res;
            }
        }

        const auto& dependencies_1 = node1->get_control_dependencies();
        const auto& dependencies_2 = node2->get_control_dependencies();

        if (dependencies_1.size() != dependencies_2.size()) {
            return error(
                "Number of dependencies is different: " + to_str(dependencies_1.size()) + " for " +
                name(node1) + " and " + to_str(dependencies_2.size()) + " for " + name(node2));
        }

        if (node1->inputs().size() != node2->inputs().size()) {
            return error(
                "Number of inputs is different: " + to_str(node1->inputs().size()) + " for " +
                name(node1) + " and " + to_str(node2->inputs().size()) + " for " + name(node2));
        }

        if (node1->outputs().size() != node2->outputs().size()) {
            return error(
                "Number of outputs is different: " + to_str(node1->inputs().size()) + " for " +
                name(node1) + " and " + to_str(node2->inputs().size()) + " for " + name(node2));
        }

        for (int i = 0; i < node1->inputs().size(); ++i) {
            if (compareConstValues) {
                using Constant = ngraph::opset1::Constant;
                auto const1 = ngraph::as_type_ptr<Constant>(node1->get_input_node_shared_ptr(i));
                auto const2 = ngraph::as_type_ptr<Constant>(node2->get_input_node_shared_ptr(i));

                if (const1 && const2 && !equal(*const1, *const2)) {
                    err_log << "Different Constant values detected\n"
                            << node1->description() << " Input(" << i << ") and "
                            << node2->description() << " Input(" << i << ")" << std::endl;
                }
            }

            if (comparePrecisions) {
                if (node1->input(i).get_element_type() != node2->input(i).get_element_type()) {
                    err_log << "Different element type detected\n"
                            << name(node1) << " Input(" << i << ") "
                            << node1->input(i).get_element_type() << " and " << name(node2)
                            << " Input(" << i << ") " << node2->input(i).get_element_type()
                            << std::endl;
                }
            }

            if (!node1->input(i).get_partial_shape().same_scheme(
                    node2->input(i).get_partial_shape())) {
                err_log << "Different shape detected\n"
                        << name(node1) << " Input(" << i << ") "
                        << node1->input(i).get_partial_shape() << " and " << name(node2)
                        << " Input(" << i << ") " << node2->input(i).get_partial_shape()
                        << std::endl;
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

            if (compareRuntimeKeys && !compare_rt_keys(node1, node2)) {
                err_log << "Different runtime info detected\n"
                        << name(node1) << " and " << name(node2) << " not equal runtime info."
                        << std::endl;
            }

            q.push(
                {node1->input_value(i).get_node_shared_ptr(),
                 node2->input_value(i).get_node_shared_ptr()});
        }

        for (int i = 0; i < node1->outputs().size(); ++i) {
            if (!node1->output(i).get_partial_shape().same_scheme(
                    node2->output(i).get_partial_shape())) {
                err_log << "Different shape detected\n"
                        << name(node1) << " Output(" << i << ") "
                        << node1->output(i).get_partial_shape() << " and " << name(node2)
                        << " Output(" << i << ") " << node2->output(i).get_partial_shape()
                        << std::endl;
            }
        }
    }

    return {err_log.str().empty(), err_log.str()};
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
