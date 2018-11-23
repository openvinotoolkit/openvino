/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "apply_adam_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id apply_adam_type_id()
{
    static primitive_type_base<apply_adam> instance;
    return &instance;
}

layout apply_adam_inst::calc_output_layout(apply_adam_node const& node)
{
    return node.input().get_non_padded_output_layout();
}

std::string apply_adam_inst::to_string(apply_adam_node const& node)
{
    auto desc      = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& m     = node.m();
    auto& v     = node.v();
    auto& beta1_power = node.beta1_power();
    auto& beta2_power = node.beta2_power();

    std::stringstream primitive_description;

    json_composite apply_adam_info;
    apply_adam_info.add("m_id", m.id());
    apply_adam_info.add("v_id", v.id());
    apply_adam_info.add("beta1_power_id", beta1_power.id());
    apply_adam_info.add("beta2_power_id", beta2_power.id());
    apply_adam_info.add("lr", desc->lr);
    apply_adam_info.add("beta1", desc->beta1);
    apply_adam_info.add("beta2", desc->beta2);
    apply_adam_info.add("epsilon", desc->epsilon);

    node_info->add("apply adam info", apply_adam_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

apply_adam_inst::typed_primitive_inst(network_impl& network, apply_adam_node const& node)
    :parent(network, node) 
{
    auto m_format = node.m().get_output_layout().format;
    auto v_format = node.v().get_output_layout().format;
    auto beta1_power_format = node.beta1_power().get_output_layout().format;
    auto beta2_power_format = node.beta2_power().get_output_layout().format;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "M format", m_format.value, "supported m formats", format::yxfb, format::bfyx );
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "V format", v_format.value, "supported v formats", format::yxfb, format::bfyx );
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "beta1_power format", beta1_power_format.value, "supported beta1_power formats", format::yxfb, format::bfyx);
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "beta2_power format", beta2_power_format.value, "supported beta2_power formats", format::yxfb, format::bfyx);
}
}