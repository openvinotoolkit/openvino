// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col_to_im_inst.h"
#include "col2im_shape_inference.hpp"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(col_to_im)

layout col_to_im_inst::calc_output_layout(col_to_im_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<col_to_im>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    // TODO : do sth here for col2im.(Copied dummy from depth_to_space)
    auto out_size = input_layout.get_tensor();
    if (format::spatial_num(input_layout.format) == 3) {
        // const size_t feature = input_layout.feature() / block_size / block_size / block_size;
        // const size_t z = input_layout.spatial(2) * block_size;
        // const size_t y = input_layout.spatial(1) * block_size;
        // const size_t x = input_layout.spatial(0) * block_size;
        // out_size = tensor(TensorValue(input_layout.batch()), TensorValue(feature), TensorValue(x), TensorValue(y), TensorValue(z));
    } else {
        // const size_t feature = input_layout.feature() / block_size / block_size;
        // const size_t y = input_layout.spatial(1) * block_size;
        // const size_t x = input_layout.spatial(0) * block_size;
        // out_size = tensor(TensorValue(input_layout.batch()), TensorValue(feature), TensorValue(x), TensorValue(y));
    }

    if (impl_param.has_fused_primitives()) {
        input_layout.data_type = impl_param.get_output_element_type();
    }

    return layout{input_layout.data_type, input_format, out_size};
}

template<typename ShapeType>
std::vector<layout> col_to_im_inst::calc_output_layouts(col_to_im_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<col_to_im>();
    auto input_layout = impl_param.get_input_layout(0);
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    ov::op::v15::Col2Im op;

    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes = ov::op::v15::shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> col_to_im_inst::calc_output_layouts<ov::PartialShape>(col_to_im_node const& node, const kernel_impl_params& impl_param);

std::string col_to_im_inst::to_string(col_to_im_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    auto strd = desc->stride;

    std::stringstream primitive_description;

    json_composite col_to_im_info;
    col_to_im_info.add("input id", input.id());
    col_to_im_info.add("stride", cldnn::to_string(strd));
    col_to_im_info.add("dilation", cldnn::to_string(desc->dilation));
    col_to_im_info.add("padding begin", cldnn::to_string(desc->padding_begin));
    col_to_im_info.add("padding end", cldnn::to_string(desc->padding_end));

    node_info->add("col_to_im info", col_to_im_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

col_to_im_inst::typed_primitive_inst(network& network, col_to_im_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
