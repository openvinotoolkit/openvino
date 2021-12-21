// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <vector>
#include <string>

namespace cldnn {
primitive_type_id reduce::type_id() {
    static primitive_type_base<reduce> instance;
    return &instance;
}

layout reduce_inst::calc_output_layout(reduce_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto input_format = input_layout.format;
    auto format_dim = input_format.dimension();
    auto output_type = input_layout.data_type;
    auto mode = desc->mode;
    auto reduce_axes = desc->axes;
    auto in_dims = input_layout.size.sizes();

    for (size_t a = 0; a < reduce_axes.size(); a++) {
        in_dims[reduce_axes[a]] = 1;
    }

    std::vector<int32_t> updated_dims;
    if (!desc->keep_dims) {
        // Get unreduced from b-f and x-w range
        for (size_t b_f_index = 0; b_f_index < 2; b_f_index++) {
            bool index_to_remove = std::find(reduce_axes.begin(), reduce_axes.end(), b_f_index) != reduce_axes.end();
            if (!index_to_remove)
                updated_dims.push_back(in_dims[b_f_index]);
        }
        for (size_t x_w_index = format_dim - 1; x_w_index >= 2; x_w_index--) {
            bool index_to_remove = std::find(reduce_axes.begin(), reduce_axes.end(), x_w_index) != reduce_axes.end();
            if (!index_to_remove)
                updated_dims.push_back(in_dims[x_w_index]);
        }

        if (input_format.dimension() == 4 && reduce_axes.size() == 1)
            updated_dims.push_back(1);
        if (updated_dims.size() > 2)
            std::reverse(updated_dims.begin() + 2, updated_dims.end());

        // Fill updated dims to format_dim size
        while (updated_dims.size() < format_dim)
            updated_dims.push_back(1);

        in_dims = std::move(updated_dims);
    }

    std::vector<reduce_mode> reduce_bool_modes = {reduce_mode::logical_and, reduce_mode::logical_or};
    if (std::find(reduce_bool_modes.begin(), reduce_bool_modes.end(), mode) != reduce_bool_modes.end())
        output_type = data_types::i8;
    else if (output_type == data_types::i8 || output_type == data_types::u8)
        output_type = data_types::f32;

    if (desc->output_data_type)
        output_type = *desc->output_data_type;

    if (node.has_fused_primitives())
        output_type = node.get_fused_output_layout().data_type;

    if (format_dim == 6)
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4], in_dims[5]))};
    else if (format_dim == 5)
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4]))};
    else
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3]))};
}

std::string reduce_inst::to_string(reduce_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite reduce_info;
    reduce_info.add("input id", node.input(0).id());
    reduce_info.add("axes", desc->axes);
    reduce_info.add("keep_dims", desc->keep_dims);
    reduce_info.add("mode", static_cast<uint16_t>(desc->mode));

    node_info->add("reduce info", reduce_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

reduce_inst::typed_primitive_inst(network& network, reduce_node const& node) : parent(network, node) {}

}  // namespace cldnn
