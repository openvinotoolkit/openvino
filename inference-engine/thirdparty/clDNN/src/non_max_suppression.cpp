/*
// Copyright (c) 2019 Intel Corporation
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
#include "non_max_suppression_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id non_max_suppression::type_id() {
    static primitive_type_base<non_max_suppression> instance;
    return &instance;
}

layout non_max_suppression_inst::calc_output_layout(non_max_suppression_node const& node) {
    auto desc = node.get_primitive();

    auto output_type = desc->output_data_type ? *desc->output_data_type : data_types::i32;

    auto output_size = tensor(batch(desc->selected_indices_num), feature(3));

    return layout(output_type, format::bfyx, output_size);
}

std::string non_max_suppression_inst::to_string(non_max_suppression_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite info;
    info.add("center point box", desc->center_point_box);

    node_info->add("non max supression info", info);

    std::stringstream description;
    node_info->dump(description);
    return description.str();
}

}  // namespace cldnn
