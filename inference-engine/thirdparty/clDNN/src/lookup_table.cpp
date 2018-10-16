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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "lookup_table_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
    primitive_type_id lookup_table_type_id()
    {
        static primitive_type_base<lookup_table> instance;
        return &instance;
    }

    layout lookup_table_inst::calc_output_layout(lookup_table_node const& node)
    {
        auto desc = node.get_primitive();

        auto input_data_layout = node.input().get_output_layout();
        auto input_indices_layout = node.indices().get_output_layout();

        return layout{ input_data_layout.data_type, input_data_layout.format, input_indices_layout.size };
    }

    std::string lookup_table_inst::to_string(lookup_table_node const& node)
    {
        auto desc = node.get_primitive();
        auto node_info = node.desc_to_json();
        auto axis = desc->with_axis ? "true" : "false";

        std::stringstream primitive_description;

        json_composite conv_info;
        conv_info.add("with axis", axis);
        if (desc->with_axis)
            conv_info.add("axis", desc->axis);
        node_info.add("lookup_table info", conv_info);
        node_info.dump(primitive_description);

        return primitive_description.str();
    }

    lookup_table_inst::typed_primitive_inst(network_impl& network, lookup_table_node const& node)
        : parent(network, node)
    {
    }
}
