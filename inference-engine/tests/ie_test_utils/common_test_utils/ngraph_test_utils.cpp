// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <queue>
#include <assert.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include "ngraph_test_utils.hpp"

bool compare(const std::vector<float>& expectedValues, const std::shared_ptr<ngraph::opset1::Constant>& constant) {
    const auto actualValues = constant->cast_vector<float>();
    if (actualValues.size() != expectedValues.size()) {
        return false;
    }

    static const float threshold = 1e-4f;
    for (size_t i = 0; i < expectedValues.size(); ++i) {
        if (abs(expectedValues[i] - actualValues[i]) > threshold) {
            return false;
        }
    }

    return true;
}

bool isTypeRelaxed(const std::string& type) {
    return type.find_first_of("TypeRelaxed") == 0;
}

bool compareTypeInfo(const ngraph::DiscreteTypeInfo& info1, const ngraph::DiscreteTypeInfo& info2) {
    if (!isTypeRelaxed(info1.name) && !isTypeRelaxed(info2.name) && (info1.version != info2.version)) {
        return false;
    }

    const std::string info1Name = isTypeRelaxed(info1.name) && (info1.parent != nullptr) ? info1.parent->name : info1.name;
    const std::string info2Name = isTypeRelaxed(info2.name) && (info2.parent != nullptr) ? info2.parent->name : info2.name;
    return info1Name == info2Name;
}

bool compare_rt_keys(const std::shared_ptr<ngraph::Node>& node1, const std::shared_ptr<ngraph::Node>& node2) {
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

    const auto& f1_results = f1->get_results();
    const auto& f2_results = f2->get_results();
    if (f1_results.size() != f2_results.size()) {
        return { false, "Number of results is different: " + std::to_string(f1_results.size()) + " and " + std::to_string(f2_results.size()) };
    }

    const auto& f1_sinks = f1->get_sinks();
    const auto& f2_sinks = f2->get_sinks();
    if (f1_sinks.size() != f2_sinks.size()) {
        return { false, "Number of sinks is different: " + std::to_string(f1_sinks.size()) + " and " + std::to_string(f2_sinks.size()) };
    }

    auto typeInfoToStr = [](const ngraph::Node::type_info_t & typeInfo) {
        return std::string(typeInfo.name) + "/" + std::to_string(typeInfo.version);
    };

    std::ostringstream err_log;

    std::queue<std::pair<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>> q;
    for (size_t i = 0; i < f1_results.size(); ++i) {
        if (compareNames) {
            if (f1_results[i]->get_input_node_shared_ptr(0)->get_friendly_name() !=
                f2_results[i]->get_input_node_shared_ptr(0)->get_friendly_name()) {
                return { false, "Different output names: " + f1_results[i]->get_input_node_shared_ptr(0)->get_friendly_name()
                    + " and " + f2_results[i]->get_input_node_shared_ptr(0)->get_friendly_name() };
            }
        }
        q.push({ f1_results[i], f2_results[i] });
    }

    while (!q.empty()) {
        auto node1 = q.front().first;
        auto node2 = q.front().second;
        q.pop();

        auto type_info1 = node1->get_type_info();
        auto type_info2 = node2->get_type_info();

        if (!compareTypeInfo(type_info1, type_info2)) {
            return {false, typeInfoToStr(type_info1) + " != " + typeInfoToStr(type_info2)};
        }

        const auto& dependencies_1 = node1->get_control_dependencies();
        const auto& dependencies_2 = node2->get_control_dependencies();
        if (dependencies_1.size() != dependencies_2.size()) {
            return {false, "Number of dependencies is different: " + std::to_string(dependencies_1.size()) + " for " + node1->get_friendly_name() +
                           + " and " + std::to_string(dependencies_2.size()) + " for " + node2->get_friendly_name()};
        }

        if (node1->inputs().size() != node2->inputs().size()) {
            return {false, "Number of inputs is different: " + std::to_string(node1->inputs().size()) + " for " + node1->get_friendly_name() +
                + " and " + std::to_string(node2->inputs().size()) + " for " + node2->get_friendly_name()};
        }

        if (node1->outputs().size() != node2->outputs().size()) {
            return { false, "Number of outputs is different: " + std::to_string(node1->inputs().size()) + " for " + node1->get_friendly_name() +
                + " and " + std::to_string(node2->inputs().size()) + " for " + node2->get_friendly_name()};
        }

        for (int i = 0; i < node1->inputs().size(); ++i) {
            if (compareConstValues) {
                std::shared_ptr<ngraph::opset1::Constant> const1 = ngraph::as_type_ptr<ngraph::opset1::Constant>(node1->get_input_node_shared_ptr(i));
                std::shared_ptr<ngraph::opset1::Constant> const2 = ngraph::as_type_ptr<ngraph::opset1::Constant>(node2->get_input_node_shared_ptr(i));
                if ((const1 != nullptr) && (const2 != nullptr)) {
                    if (!compare(const1->cast_vector<float>(), const2)) {
                        err_log << "Different Constant values detected" << std::endl
                            << node1->description() << " Input(" << i << ") and "
                            << node2->description() << " Input(" << i << ")" << std::endl;
                    }
                }
            }

            if (comparePrecisions) {
                if (node1->input(i).get_element_type() != node2->input(i).get_element_type()) {
                    err_log << "Different element type detected" << std::endl
                        << node1->get_friendly_name() << " Input(" << i << ") " << node1->input(i).get_element_type() << " and "
                        << node2->get_friendly_name() << " Input(" << i << ") " << node2->input(i).get_element_type() << std::endl;
                }
            }

            if (!node1->input(i).get_partial_shape().same_scheme(node2->input(i).get_partial_shape())) {
                err_log << "Different shape detected" << std::endl
                        << node1->get_friendly_name() << " Input(" << i << ") " << node1->input(i).get_partial_shape() << " and "
                        << node2->get_friendly_name() << " Input(" << i << ") " << node2->input(i).get_partial_shape() << std::endl;
            }

            if (node1->get_input_source_output(i).get_index() != node2->get_input_source_output(i).get_index()) {
                auto idx1 = node1->get_input_source_output(i).get_index();
                auto idx2 = node2->get_input_source_output(i).get_index();
                err_log << "Different ports detected" << std::endl
                        << node1->get_friendly_name() << " Input(" << i << ") connected to parent port " << idx1 << " and "
                        << node2->get_friendly_name() << " Input(" << i << ") connected to parent port " << idx2 << std::endl;
            }

            if (compareRuntimeKeys && !compare_rt_keys(node1, node2)) {
                err_log << "Different runtime info detected" << std::endl
                    << node1->get_friendly_name() << " and " << node2->get_friendly_name() << " not equal runttime info." << std::endl;;
            }

            q.push({node1->input_value(i).get_node_shared_ptr(), node2->input_value(i).get_node_shared_ptr()});
        }

        for (int i = 0; i < node1->outputs().size(); ++i) {
            if (!node1->output(i).get_partial_shape().same_scheme(node2->output(i).get_partial_shape())) {
                err_log << "Different shape detected" << std::endl
                        << node1->get_friendly_name() << " Output(" << i << ") " << node1->output(i).get_partial_shape() << " and "
                        << node2->get_friendly_name() << " Output(" << i << ") " << node2->output(i).get_partial_shape() << std::endl;
            }
        }
    }
    return {err_log.str().empty(), err_log.str()};
}

void check_rt_info(const std::shared_ptr<ngraph::Function> & f) {
    static const std::vector<std::string> attrs_to_check{"Variant::RuntimeAttribute::FusedNames"};

    std::ostringstream err_log;
    for (auto & op : f->get_ops()) {
        if (ngraph::op::is_constant(op)) continue;

        const auto & rt_info = op->get_rt_info();
        for (const auto & attr_name : attrs_to_check) {
            if (!rt_info.count(attr_name)) {
                err_log << "Node: " << op->get_friendly_name() << " has no attribute: " << attr_name << std::endl;
            }
        }
    }

    auto err_msg = err_log.str();
    if (!err_msg.empty()) {
        throw ngraph::ngraph_error(err_msg);
    }
}

NGRAPH_RTTI_DEFINITION(TestOpMultiOut, "TestOp", 0);
