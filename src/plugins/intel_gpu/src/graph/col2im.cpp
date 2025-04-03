// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im_inst.h"
#include "col2im_shape_inference.hpp"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(col2im)

bool col2im_inst::validate_num_blocks(kernel_impl_params const& impl_param, size_t candidate_num_blocks) {
    constexpr size_t spatial_dims = 2;
    auto desc = impl_param.typed_desc<col2im>();

    size_t L_calculated = 1;
    for (size_t d = 0; d < spatial_dims; ++d) {
        L_calculated *= ((desc->output_shape[d] + desc->padding_begin[d] + desc->padding_end[d] -
                                (desc->dilation[d] * (desc->kernel_shape[d] - 1)) - 1) / desc->stride[d]) + 1;
    }

    return (candidate_num_blocks == L_calculated);
}

layout col2im_inst::calc_output_layout(col2im_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<col2im>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    bool is_batched = true;
    auto num_blocks_l = input_layout.spatial(1);
    if (num_blocks_l == 1 && !validate_num_blocks(impl_param, input_layout.spatial(1)))
        is_batched = false;

    auto out_tensor = input_layout.get_tensor();
    auto num_elements = is_batched ? input_layout.feature() : input_layout.batch();

    const size_t batch = is_batched ? input_layout.batch() : 1;
    const size_t feature = std::max(num_elements / (desc->kernel_shape[0] * desc->kernel_shape[1]), (size_t)1);
    const size_t y = desc->output_shape[1];
    const size_t x = desc->output_shape[0];

    if (format::spatial_num(input_layout.format) == 3) {
        const size_t z = 1;
        out_tensor = tensor(TensorValue(batch), TensorValue(feature), TensorValue(x), TensorValue(y), TensorValue(z));
    } else {
        out_tensor = tensor(TensorValue(batch), TensorValue(feature), TensorValue(x), TensorValue(y));
    }

    if (impl_param.has_fused_primitives()) {
        input_layout.data_type = impl_param.get_output_element_type();
    }

    return layout{input_layout.data_type, input_format, out_tensor};
}

template<typename ShapeType>
std::vector<layout> col2im_inst::calc_output_layouts(col2im_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<col2im>();
    auto input_layout = impl_param.get_input_layout(0);
    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    ov::op::v15::Col2Im op;
    op.set_strides(desc->stride);
    op.set_dilations(desc->dilation);
    op.set_pads_begin(ov::Shape(desc->padding_begin.begin(), desc->padding_begin.end()));
    op.set_pads_end(ov::Shape(desc->padding_end.begin(), desc->padding_end.end()));

    // output_size is 1D tensor of two positive integer numbers (height and width).
    std::vector<size_t> output_size = {desc->output_shape[0], desc->output_shape[1]};
    // kernel_size is 1D tensor of non-negative integer numbers
    std::vector<size_t> kernel_size = {desc->kernel_shape[0], desc->kernel_shape[1]};

    auto output_tensor = ov::Tensor(ov::element::Type_t::u64, ov::Shape{ output_size.size() }, output_size.data());
    auto kernel_tensor = ov::Tensor(ov::element::Type_t::u64, ov::Shape{ kernel_size.size() }, kernel_size.data());

    std::unordered_map<size_t, ov::Tensor> const_data;
    const_data.emplace(1, output_tensor);
    const_data.emplace(2, kernel_tensor);

    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        output_tensor.get_shape(),
        kernel_tensor.get_shape(),
    };

    std::vector<ShapeType> output_shapes;
    output_shapes = ov::op::v15::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> col2im_inst::calc_output_layouts<ov::PartialShape>(col2im_node const& node, const kernel_impl_params& impl_param);

std::string col2im_inst::to_string(col2im_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    auto strd = desc->stride;

    std::stringstream primitive_description;

    json_composite col2im_info;
    col2im_info.add("input id", input.id());
    col2im_info.add("stride", cldnn::to_string(strd));
    col2im_info.add("dilation", cldnn::to_string(desc->dilation));
    col2im_info.add("padding begin", cldnn::to_string(desc->padding_begin));
    col2im_info.add("padding end", cldnn::to_string(desc->padding_end));

    node_info->add("col2im info", col2im_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

col2im_inst::typed_primitive_inst(network& network, col2im_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
