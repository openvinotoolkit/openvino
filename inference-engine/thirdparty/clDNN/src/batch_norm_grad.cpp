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

#include "batch_norm_grad_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
    primitive_type_id batch_norm_grad_type_id()
    {
        static primitive_type_base<batch_norm_grad> instance;
        return &instance;
    }

    layout batch_norm_grad_inst::calc_output_layout(parent::typed_node const& node)
    {
        return node.input().get_non_padded_output_layout();
    }

    std::string batch_norm_grad_inst::to_string(batch_norm_grad_node const& node)
    {
        auto desc = node.get_primitive();
        auto node_info = node.desc_to_json();
        auto& inv_var = node.inv_variance();

        std::stringstream primitive_description;

        json_composite batch_norm_grad_info;

        batch_norm_grad_info.add("inv_variance_id", inv_var.id());

        node_info.add("batch_norm_grad info", batch_norm_grad_info);
        node_info.dump(primitive_description);

        return primitive_description.str();
    }

    batch_norm_grad_inst::typed_primitive_inst(network_impl& network, batch_norm_grad_node const& node)
        :parent(network, node)
    {
    }

}
