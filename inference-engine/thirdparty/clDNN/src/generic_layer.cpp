/*
// Copyright (c) 2016 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "generic_layer_inst.h"
#include "primitive_type_base.h"

#include "json_object.h"

#include <algorithm>

namespace cldnn
{

primitive_type_id generic_layer_type_id()
{
    static primitive_type_base<generic_layer> instance;
    return &instance;
}

generic_layer_inst::typed_primitive_inst(network_impl& network, generic_layer_node const& node)
    : parent(network, node)
{
}

std::string generic_layer_inst::to_string(generic_layer_node const& node)
{
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

}
