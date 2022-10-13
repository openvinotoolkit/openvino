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

layout non_max_suppression_inst::calc_output_layout(non_max_suppression_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<non_max_suppression>();

    auto output_type = desc->output_data_type ? *desc->output_data_type : data_types::i32;

    auto output_size = tensor(batch(desc->selected_indices_num), feature(3));
    return layout(output_type, impl_param.get_input_layout().format, output_size);
}

template<typename ShapeType>
std::vector<layout> non_max_suppression_inst::calc_output_layouts(non_max_suppression_node const& /*node*/, const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;
    auto desc = impl_param.typed_desc<non_max_suppression>();

    for (size_t i = 0; i < desc->num_outputs; ++i) {
        auto output_type = desc->output_data_type ? *desc->output_data_type : data_types::i32;
        auto output_size = tensor(batch(desc->selected_indices_num), feature(3));
        if (i == 1) // selected_scores
            output_type = impl_param.get_input_layout(1).data_type;
        else if (i == 2) // valid_outputs
            output_size = tensor(batch(1));
        layouts.push_back(layout(output_type, impl_param.get_input_layout().format, output_size));
    }
    return layouts;
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
