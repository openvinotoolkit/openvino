// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(gather_nd)

layout gather_nd_inst::calc_output_layout(gather_nd_node const& node, kernel_impl_params const& impl_param) {
    auto op = impl_param.typed_desc<gather_nd>();

    auto input_layout_origin = impl_param.get_input_layout(0);
    auto indices_layout_origin = impl_param.get_input_layout(1);

    auto input_layout = input_layout_origin.get_tensor().sizes(input_layout_origin.format);
    auto indices_layout = indices_layout_origin.get_tensor().sizes(indices_layout_origin.format);

    const auto input_rank = static_cast<size_t>(op->input_rank);
    const auto indices_rank = op->indices_rank;
    const auto batch_dims = op->batch_dims;

    // calculate initial output shape
    std::vector<tensor::value_type> output_sizes;

    for (uint8_t x = 0; x < indices_rank - 1; x++) {
        output_sizes.push_back(indices_layout[x]);
    }

    const size_t indices_last_dim = indices_layout[indices_rank - 1];
    for (size_t x = static_cast<size_t>(batch_dims + indices_last_dim); x < input_rank; x++) {
        output_sizes.push_back(input_layout[x]);
    }

    // create final output shape by batch_dims
    std::vector<tensor::value_type> final_output_sizes;

    if (op->batch_merged_output) {
        // calculate batch_size by batch_dims
        int batch_size = 1;
        for (uint8_t x = 0; x < batch_dims; x++) {
            batch_size *= output_sizes[x];
        }

        if (batch_dims > 0) {
            final_output_sizes.push_back(batch_size);
        }

        for (size_t x = static_cast<size_t>(batch_dims); x < output_sizes.size(); x++) {
            final_output_sizes.push_back(output_sizes[x]);
        }
    } else {
        for (size_t x = 0; x < output_sizes.size(); x++) {
            final_output_sizes.push_back(output_sizes[x]);
        }
    }

    auto output_format = cldnn::format::any;
    if (final_output_sizes.size() <= 4) {
        output_format = cldnn::format::bfyx;
    } else if (final_output_sizes.size() == 5) {
        output_format = cldnn::format::bfzyx;
    } else {
        output_format = cldnn::format::bfwzyx;
    }

    auto output_sizes_tensor = tensor(tensor(final_output_sizes).sizes(output_format));
    auto padding = op->output_paddings[0];

    if (impl_param.has_fused_primitives()) {
        input_layout_origin.data_type = impl_param.get_fused_output_layout().data_type;
    }

    return layout(input_layout_origin.data_type, output_format, output_sizes_tensor, padding);
}

std::string gather_nd_inst::to_string(gather_nd_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_nd_info;
    gather_nd_info.add("input id", input.id());
    gather_nd_info.add("indices rank", desc->indices_rank);
    gather_nd_info.add("batch dims", desc->batch_dims);

    node_info->add("gather_nd info", gather_nd_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_nd_inst::typed_primitive_inst(network& network, gather_nd_node const& node) : parent(network, node) {}

}  // namespace cldnn
