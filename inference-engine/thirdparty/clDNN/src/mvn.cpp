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

#include "mvn_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id mvn_type_id()
{
    static primitive_type_base<mvn> instance;
    return &instance;
}

layout mvn_inst::calc_output_layout(mvn_node const& node)
{
    assert((bool)node.get_primitive()->output_data_type == false
           && "Output data type forcing is not supported for mvn_node!");
    return node.input().get_non_padded_output_layout();
}

std::string mvn_inst::to_string(mvn_node const& node)
{
    auto node_info    = node.desc_to_json();
    auto desc         = node.get_primitive();
    auto epsilon      = desc->epsilon;
    auto across_channels = desc->across_channels ? "true" : "false";
    auto normalize_variance = desc->normalize_variance ? "true" : "false";
    auto& input       = node.input();
    
    std::stringstream primitive_description;

    json_composite mvn_info;
    mvn_info.add("input id", input.id());
    mvn_info.add("epsilon", epsilon);
    mvn_info.add("across_channels region", across_channels);
    mvn_info.add("normalize_variance region", normalize_variance);

    node_info->add("mvn info", mvn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

mvn_inst::typed_primitive_inst(network_impl& network, mvn_node const& node)
    :parent(network, node)
{

}
}
