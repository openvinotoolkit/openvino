// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/specialize_function.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> specialize_function(std::shared_ptr<ov::Model> model,
                                               const std::vector<ov::element::Type>& parameter_element_types,
                                               const std::vector<ov::PartialShape>& parameter_shapes,
                                               const std::vector<void*>& parameter_values) {
    OPENVINO_ASSERT(model->get_parameters().size() == parameter_shapes.size());
    OPENVINO_ASSERT(model->get_parameters().size() == parameter_element_types.size());
    OPENVINO_ASSERT(model->get_parameters().size() == parameter_values.size());

    std::unordered_map<Node*, std::shared_ptr<Node>> nodes;

    for (size_t i = 0; i < parameter_shapes.size(); i++) {
        OPENVINO_ASSERT(model->get_parameters()[i]->get_element_type().is_dynamic() ||
                        parameter_element_types[i] == model->get_parameters()[i]->get_element_type());

        if (parameter_values[i] != nullptr && parameter_shapes[i].is_static() &&
            parameter_element_types[i].is_static()) {
            nodes[model->get_parameters()[i].get()] = std::make_shared<op::v0::Constant>(parameter_element_types[i],
                                                                                         parameter_shapes[i].to_shape(),
                                                                                         parameter_values[i]);
        } else {
            nodes[model->get_parameters()[i].get()] =
                std::make_shared<op::v0::Parameter>(parameter_element_types[i], parameter_shapes[i]);
        }
        auto rt_info = model->get_parameters()[i]->get_rt_info();
        nodes[model->get_parameters()[i].get()]->get_rt_info() = rt_info;
    }

    for (auto old_node : model->get_ordered_ops()) {
        if (op::util::is_parameter(old_node)) {
            continue;
        }

        OutputVector new_args;
        for (auto input : old_node->inputs()) {
            auto output = input.get_source_output();
            new_args.push_back(output.for_node(nodes[output.get_node()]));
        }

        NodeVector cloned_dependencies;
        for (auto& dependency : old_node->get_control_dependencies()) {
            std::shared_ptr<Node> dependent = nodes.at(dependency.get());
            if (find(cloned_dependencies.begin(), cloned_dependencies.end(), dependent) == cloned_dependencies.end()) {
                cloned_dependencies.push_back(dependent);
            }
        }
        nodes[old_node.get()] = old_node->copy_with_new_inputs(new_args, cloned_dependencies);

        auto rt_info = old_node->get_rt_info();
        nodes[old_node.get()]->get_rt_info() = rt_info;

        nodes[old_node.get()]->set_friendly_name(old_node->get_friendly_name());
    }

    ParameterVector new_parameters = model->get_parameters();
    for (size_t i = 0; i < new_parameters.size(); i++) {
        auto name = new_parameters[i]->get_friendly_name();
        new_parameters[i] = as_type_ptr<op::v0::Parameter>(nodes[new_parameters[i].get()]);

        // If the replacement for a Parameter is not itself a Parameter, we must have replaced it
        // with a constant. We will insert a dead Parameter into the clone's parameters, in order
        // to maintain the arity of the original function.
        if (new_parameters[i] == nullptr) {
            new_parameters[i] = std::make_shared<op::v0::Parameter>(parameter_element_types[i], parameter_shapes[i]);
        }
        new_parameters[i]->set_friendly_name(name);
    }

    ResultVector new_results = model->get_results();
    for (size_t i = 0; i < new_results.size(); i++) {
        auto name = new_results[i]->get_friendly_name();
        new_results[i] = std::static_pointer_cast<op::v0::Result>(nodes[new_results[i].get()]);
        new_results[i]->set_friendly_name(name);
    }
    auto new_sinks = model->get_sinks();
    for (size_t i = 0; i < new_sinks.size(); i++) {
        new_sinks[i] = std::static_pointer_cast<op::Sink>(nodes[new_sinks[i].get()]);
    }

    return std::make_shared<Model>(new_results, new_sinks, new_parameters);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
