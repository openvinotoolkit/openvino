// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/preprocessing/subtract_mean_inputs.hpp"

using namespace ngraph;
using namespace ngraph::pass;

NGRAPH_RTTI_DEFINITION(ngraph::pass::SubtractMeanInputs, "SubtractMeanInputs", 0);

SubtractMeanInputs::SubtractMeanInputs(const MeanMap& mean_map):
m_mean_map(mean_map) {}

bool SubtractMeanInputs::run_on_function(std::shared_ptr<ngraph::Function> function) {
    bool updated = false;
    for (auto param : function->get_parameters()) {
        auto consumers = param->output(0).get_target_inputs();
        auto it = m_mean_map.find(param->get_friendly_name());
        if (it != m_mean_map.end()) {
            auto constant = it->second;
            NGRAPH_CHECK(constant->get_element_type() == ngraph::element::f32,
                         "Subtract mean for ", param->get_friendly_name(), " must have f32 type");
            auto constant_copy = std::make_shared<ngraph::op::Constant>(*constant);
            constant_copy->set_friendly_name(param->get_friendly_name() + "/subtract/SubtractMean_Value");

            auto new_op = std::make_shared<ngraph::op::v1::Subtract>(param, constant_copy);
            new_op->set_friendly_name(param->get_friendly_name() + "/subtract/SubtractMean");
            for (auto consumer : consumers) {
                consumer.replace_source_output(new_op);
            }
            updated = true;
        }
    }
    return updated;
}
