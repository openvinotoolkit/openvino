// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <limits>

#include "topk_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(arg_max_min)

layout arg_max_min_inst::calc_output_layout(arg_max_min_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<arg_max_min>();
    auto input_layout = impl_param.get_input_layout();
    bool values_first = desc->values_first;
    data_types output_data_type;
    data_types output_idx_type;
    output_data_type = desc->output_data_types[0].value_or(input_layout.data_type);
    if (impl_param.input_layouts.size() == 3) {
        output_idx_type = impl_param.get_input_layout(2).data_type;
    } else {
        output_idx_type = *(desc->output_data_types[0]);
    }
    auto size_check = [&](size_t tensor_size) {
        if (desc->input.size() == 1 && values_first)
            return;
        size_t max_size;
        // lowest integer not representable in floating point type = 2^(mantissa_bits + 1) + 1
        // https://stackoverflow.com/questions/3793838/which-is-the-first-integer-that-an-ieee-754-float-is-incapable-of-representing-e
        if (output_idx_type == data_types::f32) {
            max_size = (1 << std::numeric_limits<float>::digits);
        } else if (output_idx_type == data_types::f16) {
            // mantissa_bits for fp16 = 10
            max_size = (1 << 11);
        } else if (output_idx_type == data_types::u8) {
            max_size = std::numeric_limits<uint8_t>::max();
        } else if (output_idx_type == data_types::i32) {
            max_size = std::numeric_limits<int32_t>::max();
        } else {
            max_size = std::numeric_limits<size_t>::max();
        }

        if (tensor_size > max_size) {
            CLDNN_ERROR_GREATER_THAN(desc->id,
                                     "Reduced tensor size",
                                     tensor_size,
                                     "Maximum output data type value",
                                     max_size,
                                     "Current output data type is unable to hold maximum index of a tensor.");
        }
    };
    for (auto dim : input_layout.get_dims()) {
        size_check(dim);
    }
    auto format = input_layout.format;
    auto sizes = input_layout.get_dims();
    if (desc->axis >= static_cast<int64_t>(sizes.size()) || desc->axis < 0) {
        OPENVINO_THROW("Incorrect arg_max_min axis.");
    }
    sizes[desc->axis] = desc->top_k;
    return layout{output_data_type, format, tensor(format::get_default_format(input_layout.get_rank()), sizes)};
}

template<typename ShapeType>
std::vector<layout> arg_max_min_inst::calc_output_layouts(arg_max_min_node const& /*node*/, const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;

    auto desc = impl_param.typed_desc<arg_max_min>();
    auto input_layout = impl_param.get_input_layout();

    ov::op::v1::TopK op;
    auto input_rank = input_layout.get<ShapeType>().rank();
    op.set_axis(input_rank, desc->axis);
    op.set_mode(desc->mode);
    op.set_sort_type(desc->sort);

    std::vector<ShapeType> output_shapes = {ShapeType{}, ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input_layout.get<ShapeType>(),
        ShapeType{}
    };

    auto& constant_mem = impl_param.memory_deps;
    if (desc->top_k > 0) {
        std::unordered_map<size_t, ov::Tensor> const_data;
        auto topk = desc->top_k;
        auto top_k_tensor = ov::Tensor(ov::element::u32, ov::Shape{1}, static_cast<void*>(&topk));
        const_data = { {1, top_k_tensor} };

        output_shapes = ov::op::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    } else if (constant_mem.count(1)) {
        std::unordered_map<size_t, ov::Tensor> const_data;
        auto target_shape_mem = constant_mem.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> target_shape_lock(target_shape_mem, impl_param.get_stream());
        const_data.emplace(1, make_tensor(target_shape_mem->get_layout(), target_shape_lock.data()));

        output_shapes = ov::op::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    } else {
        output_shapes[0] = output_shapes[1] = ShapeType::dynamic(input_layout.get<ShapeType>().size());
    }

    for (size_t i = 0; i < desc->num_outputs; ++i) {
        auto dt = desc->get_output_data_type(i).value_or(input_layout.data_type);
        layouts.push_back({output_shapes[i], dt, format::get_default_format(output_shapes[i].size())});
    }
    return layouts;
}

template std::vector<layout> arg_max_min_inst::calc_output_layouts<ov::PartialShape>(arg_max_min_node const& node, const kernel_impl_params& impl_param);

std::string arg_max_min_inst::to_string(arg_max_min_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite conv_info;
    conv_info.add("top_k", desc->top_k);
    conv_info.add("axis", desc->axis);
    conv_info.add("output type", desc->mode);
    conv_info.add("sort type", desc->sort);
    node_info->add("arg_max_min info", conv_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

arg_max_min_inst::typed_primitive_inst(network& network, arg_max_min_node const& node) : parent(network, node) {}
}  // namespace cldnn
