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
#include "custom_gpu_primitive_inst.h"
#include "primitive_type_base.h"
#include <sstream>
#include "json_object.h"

namespace cldnn
{

primitive_type_id custom_gpu_primitive_type_id()
{
    static primitive_type_base<custom_gpu_primitive> instance;
    return &instance;
}

std::string custom_gpu_primitive_inst::to_string(custom_gpu_primitive_node const& node)
{
    auto desc      = node.get_primitive();
    auto node_info = node.desc_to_json();
    
    std::stringstream primitive_description;

    json_composite custom_gpu_prim_info;
    custom_gpu_prim_info.add("entry point",   desc->kernel_entry_point);
    custom_gpu_prim_info.add("kernels code",  desc->kernels_code.ref());
    custom_gpu_prim_info.add("build options", desc->build_options);
    custom_gpu_prim_info.add("gws", desc->gws);
    custom_gpu_prim_info.add("lws", desc->lws);
    // TODO: consider printing more information here
    node_info.add("custom primitive info", custom_gpu_prim_info);
    node_info.dump(primitive_description);

    return primitive_description.str();
}

custom_gpu_primitive_inst::typed_primitive_inst(network_impl& network, custom_gpu_primitive_node const& node)
    : parent(network, node)
{
}
}
