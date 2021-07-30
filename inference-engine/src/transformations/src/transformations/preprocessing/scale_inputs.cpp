// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/preprocessing/scale_inputs.hpp"

using namespace ngraph;
using namespace ngraph::pass;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ScaleInputs, "ScaleInputs", 0);

ScaleInputs::ScaleInputs(const ScaleMap& scale_map):
m_scale_map(scale_map) {}

bool ScaleInputs::run_on_function(std::shared_ptr<ngraph::Function> function) {
    bool updated = false;
    for (auto param : function->get_parameters()) {
        auto consumers = param->output(0).get_target_inputs();
        auto it = m_scale_map.find(param->get_friendly_name());
        if (it != m_scale_map.end()) {
            auto constant = it->second;
            NGRAPH_CHECK(constant->get_element_type() == ngraph::element::f32,
                         "Scale for ", param->get_friendly_name(), " must have f32 type");
            auto constant_copy = std::make_shared<ngraph::op::Constant>(*constant);
            constant_copy->set_friendly_name(param->get_friendly_name() + "/scale/Divide_Factor");

            auto new_op = std::make_shared<ngraph::opset1::Divide>(param, constant_copy);
            new_op->set_friendly_name(param->get_friendly_name() + "/scale/Divide");
            for (auto consumer : consumers) {
                consumer.replace_source_output(new_op);
            }
            updated = true;
        }
    }
    return updated;
}
