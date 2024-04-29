// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd_inst.h"
#include "gather_nd_shape_inference.hpp"

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

    auto output_format = format::get_default_format(final_output_sizes.size());
    auto output_sizes_tensor = tensor(tensor(final_output_sizes).sizes(output_format));
    auto padding = op->output_paddings[0];

    if (impl_param.has_fused_primitives()) {
        input_layout_origin.data_type = impl_param.get_output_element_type();
    }

    return layout(input_layout_origin.data_type, output_format, output_sizes_tensor, padding);
}


template<typename ShapeType>
std::vector<layout> gather_nd_inst::calc_output_layouts(gather_nd_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<gather_nd>();

    auto input_layout = impl_param.get_input_layout(0);
    auto indices_layout = impl_param.get_input_layout(1);

    auto output_type = input_layout.data_type;
    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    }

    std::vector<ShapeType> output_shapes;
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        indices_layout.get<ShapeType>()
    };

    if (desc->batch_merged_output) {
        ov::op::v5::GatherND op;
        op.set_batch_dims(desc->batch_dims);
        output_shapes = ov::op::v5::shape_infer(&op, input_shapes);
    } else {
        ov::op::v8::GatherND op;
        op.set_batch_dims(desc->batch_dims);
        output_shapes = ov::op::v8::shape_infer(&op, input_shapes);
    }

    OPENVINO_ASSERT(!output_shapes[0].rank().is_dynamic(),
                    "[GPU] Doesn't support output dynamic rank in gather_nd");
    format output_format = format::adjust_to_rank(input_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> gather_nd_inst::calc_output_layouts<ov::PartialShape>(gather_nd_node const& node, const kernel_impl_params& impl_param);

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
