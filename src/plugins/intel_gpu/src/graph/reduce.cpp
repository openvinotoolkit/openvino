// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <vector>
#include <string>

#include "reduce_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(reduce)

static std::vector<uint16_t> convert_axes(std::vector<int64_t> axes, size_t rank) {
    std::vector<uint16_t> converted_axes;
    for (auto axis : axes) {
        if (axis == 0 || axis == 1) {
            converted_axes.push_back(axis);
            continue;
        }

        if (axis < 0)
            axis = axis + rank;

        converted_axes.push_back(static_cast<uint16_t>(rank + 1 - axis));
    }

    return converted_axes;
}

layout reduce_inst::calc_output_layout(reduce_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<reduce>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;
    auto format_dim = input_format.dimension();
    auto output_type = input_layout.data_type;
    auto mode = desc->mode;
    auto reduce_axes = convert_axes(desc->axes, input_layout.get_rank());
    auto in_dims = input_layout.get_tensor().sizes();

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
    if (std::find(reduce_bool_modes.begin(), reduce_bool_modes.end(), mode) != reduce_bool_modes.end()) {
        output_type = data_types::i8;
    } else if (output_type == data_types::i8 || output_type == data_types::u8) {
        if (mode != reduce_mode::min && mode != reduce_mode::max) {
            output_type = data_types::f32;
        }
    }

    if (desc->output_data_types[0])
        output_type = *desc->output_data_types[0];

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    if (format_dim == 6)
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4], in_dims[5]))};
    else if (format_dim == 5)
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3], in_dims[4]))};
    else
        return layout{output_type, input_format, tensor(batch(in_dims[0]), feature(in_dims[1]), spatial(in_dims[2], in_dims[3]))};
}

template<typename ShapeType>
std::vector<layout> reduce_inst::calc_output_layouts(reduce_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<reduce>();

    auto input0_layout = impl_param.get_input_layout(0);

    // get 'output_shapes' by shape_infer of ngraph
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        ShapeType{0}
    };

    std::vector<ShapeType> output_shapes = {ShapeType()};

    auto axes = desc->axes;
    auto axes_tensor = ov::Tensor(ov::element::i64, ov::Shape{axes.size()}, axes.data());
    std::unordered_map<size_t, ov::Tensor> const_data = {{1, axes_tensor}};
    auto ta = ov::make_tensor_accessor(const_data);

    // shape infer by mode
    auto mode = desc->mode;
    auto keep_dims = desc->keep_dims;
    switch (mode) {
        case reduce_mode::max:
        {
            ov::op::v1::ReduceMax op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::min:
        {
            ov::op::v1::ReduceMin op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::mean:
        {
            ov::op::v1::ReduceMean op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::prod:
        {
            ov::op::v1::ReduceProd op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::sum:
        {
            ov::op::v1::ReduceSum op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::logical_and:
            {
            ov::op::v1::ReduceLogicalAnd op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::logical_or:
        {
            ov::op::v1::ReduceLogicalOr op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::l1:
        {
            ov::op::v4::ReduceL1 op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::l2:
        {
            ov::op::v4::ReduceL2 op;
            op.set_keep_dims(keep_dims);
            output_shapes = shape_infer(&op, input_shapes, ta);
            break;
        }
        case reduce_mode::sum_square:
            // not implemented
        case reduce_mode::log_sum:
            // not implemented
        case reduce_mode::log_sum_exp:
            // not implemented
        default:
            OPENVINO_ASSERT(false, "Not supported reduce mode");
    }

    auto input_type = input0_layout.data_type;
    auto output_type = input_type;
    std::vector<reduce_mode> reduce_bool_modes = {reduce_mode::logical_and, reduce_mode::logical_or};
    if (std::find(reduce_bool_modes.begin(), reduce_bool_modes.end(), mode) != reduce_bool_modes.end())
        output_type = data_types::i8;
    else if (input_type == data_types::i8 || input_type == data_types::u8)
        output_type = data_types::f32;

    output_type = desc->output_data_types[0].value_or(output_type);

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    auto output_format = format::adjust_to_rank(input0_layout.format, output_shapes[0].size());

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> reduce_inst::calc_output_layouts<ov::PartialShape>(reduce_node const& node, const kernel_impl_params& impl_param);

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
