// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

    auto output_type = desc->output_data_types.at(0) ? *desc->output_data_types.at(0) : data_types::i32;

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
