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

#include "softmax_loss_grad_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id softmax_loss_grad_type_id()
{
    static primitive_type_base<softmax_loss_grad> instance;
    return &instance;
}

layout softmax_loss_grad_inst::calc_output_layout(softmax_loss_grad_node const& node)
{
    assert((bool)node.get_primitive()->output_data_type == false
           && "Output data type forcing is not supported for "
              "softmax_loss_grad_node!");
    return node.input().get_non_padded_output_layout();
}

std::string softmax_loss_grad_inst::to_string(softmax_loss_grad_node const& node)
{
    auto desc      = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

softmax_loss_grad_inst::typed_primitive_inst(network_impl& network, softmax_loss_grad_node const& node)
    : parent(network, node)
{
    //TODO: add size check here for labels
}
}
