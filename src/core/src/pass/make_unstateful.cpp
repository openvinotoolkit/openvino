// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_unstateful.hpp"

#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/pass/itt.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ngraph;
using namespace opset9;
using namespace op::util;
using namespace ov::pass;

bool MakeUnStateful::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(MakeUnStateful);

    ov::ParameterVector params;
    ov::ResultVector results;

    auto sinks = f->get_sinks();
    for (auto& sink : sinks) {
        std::string parameter_name = "";
        std::string result_name = "";
        auto assign = ov::as_type_ptr<ngraph::op::AssignBase>(sink);
        auto variable = assign->get_variable();
        std::pair<std::string, std::string> param_result_names = {};
        auto it = m_variable_map.find(variable);
        if (it != m_variable_map.end()) {
            parameter_name = it->second.first;
            result_name = it->second.second;
        } else {
            auto variable_id = variable->get_info().variable_id;
            auto it = m_variable_names_map.find(variable_id);
            if (it != m_variable_names_map.end()) {
                parameter_name = it->second.first;
                result_name = it->second.second;
            } else {
                continue;
            }
        }
        for (auto& dep : sink->get_control_dependencies()) {
            auto read_value = ov::as_type_ptr<ngraph::op::ReadValueBase>(dep);
            if (read_value) {
                auto param = std::make_shared<ngraph::op::Parameter>(read_value->get_element_type(),
                                                                     read_value->input(0).get_shape());
                param->set_output_size(1);
                param->set_friendly_name(parameter_name);
                replace_node(read_value, param);
                copy_runtime_info(read_value, param);
                params.push_back(param);
                sink->remove_control_dependency(dep);
            }
            auto result = std::make_shared<ngraph::op::Result>(assign->input_value(0));
            result->set_friendly_name(result_name);
            copy_runtime_info(assign, result);
            results.push_back(result);
            f->remove_sink(sink);
            f->remove_variable(variable);
        }
    }
    f->add_parameters(params);
    f->add_results(results);
    return true;
}
