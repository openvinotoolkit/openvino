// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_batch_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

#include "space_to_batch_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(space_to_batch)

layout space_to_batch_inst::calc_output_layout(space_to_batch_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<space_to_batch>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    const size_t spatial_num = format::spatial_num(input_format);

    const auto& block_shape = desc->block_shape;
    const auto& pads_begin = desc->pads_begin;
    const auto& pads_end = desc->pads_end;

    if (block_shape.batch[0] != 1)
        CLDNN_ERROR_MESSAGE(desc->id,
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape.batch[0]));

    if (pads_begin.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "pads_begin[0] is expected to be 0. Actual pads_begin[0] is " +
            std::to_string(pads_begin.batch[0]));

    if (pads_end.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "pads_end[0] is expected to be 0. Actual pads_end[0] is " +
            std::to_string(pads_end.batch[0]));

    if ((input_layout.feature() + pads_begin.feature[0] + pads_end.feature[0]) % block_shape.feature[0] != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
                "Input feature shape after padding must be divisible by block_shape");

    for (size_t i = 0; i < spatial_num; ++i)
        if ((input_layout.spatial(i) + pads_begin.spatial[i] + pads_end.spatial[i]) % block_shape.spatial[i] != 0)
            CLDNN_ERROR_MESSAGE(desc->id,
                "Input spatial shapes after padding must be divisible by block_shape");

    return layout{output_type, input_format, desc->out_size};
}

static std::vector<int32_t> tensor_to_vec(const tensor& t, const format f) {
    std::vector<int32_t> vec(cldnn::format::dimension(f));
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = t.sizes()[i];
    }
    std::reverse(vec.begin() + 2, vec.end());
    return vec;
}

template<typename ShapeType>
std::vector<layout> space_to_batch_inst::calc_output_layouts(space_to_batch_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<space_to_batch>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto input0_shape = input0_layout.get<ShapeType>();
    auto input0_size = input0_shape.size();
    auto input0_format = input0_layout.format;

    auto& constant_mem = impl_param.memory_deps;
    auto block_data = desc->block_shape;
    auto begin_data = desc->pads_begin;
    auto end_data = desc->pads_end;

    if (desc->shape_constant == 0 && (!constant_mem.count(1) || !constant_mem.count(2) || !constant_mem.count(3))) {
        auto out_shape = ov::PartialShape::dynamic(input0_size);
        return { layout{out_shape, input0_layout.data_type, input0_format } };
    }


    ShapeType block_shape = desc->shape_constant == 0 ? impl_param.get_input_layout(1).get<ShapeType>() : ov::Shape{ input0_size };
    ShapeType begin_shape = desc->shape_constant == 0 ? impl_param.get_input_layout(2).get<ShapeType>() : ov::Shape{ input0_size };
    ShapeType end_shape = desc->shape_constant == 0 ? impl_param.get_input_layout(3).get<ShapeType>() : ov::Shape{ input0_size };

    ov::op::v1::SpaceToBatch op;
    std::vector<ShapeType> output_shapes = {ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input0_shape,
        block_shape,
        begin_shape,
        end_shape
    };

    std::unordered_map<size_t, ov::Tensor> const_data;
    if (desc->shape_constant) {
        auto block_sizes = tensor_to_vec(block_data, input0_format);
        auto begin_sizes = tensor_to_vec(begin_data, input0_format);
        auto end_sizes = tensor_to_vec(end_data, input0_format);

        auto block_values = static_cast<void*>(block_sizes.data());
        auto begin_values = static_cast<void*>(begin_sizes.data());
        auto end_values = static_cast<void*>(end_sizes.data());

        auto block_tensor = make_tensor({ block_shape, data_types::i32, input0_format }, block_values);
        auto begin_tensor = make_tensor({ begin_shape, data_types::i32, input0_format }, begin_values);
        auto end_tensor = make_tensor({ end_shape, data_types::i32, input0_format }, end_values);

        const_data.emplace(1, block_tensor);
        const_data.emplace(2, begin_tensor);
        const_data.emplace(3, end_tensor);

        output_shapes = ov::op::v1::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    } else {
        auto block_mem = constant_mem.at(1);
        auto begin_mem = constant_mem.at(2);
        auto end_mem = constant_mem.at(3);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(block_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock2(begin_mem, impl_param.get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock3(end_mem, impl_param.get_stream());

        auto block_tensor = make_tensor(block_mem->get_layout(), lock1.data());
        auto begin_tensor = make_tensor(begin_mem->get_layout(), lock2.data());
        auto end_tensor = make_tensor(end_mem->get_layout(), lock3.data());

        const_data.emplace(1, block_tensor);
        const_data.emplace(2, begin_tensor);
        const_data.emplace(3, end_tensor);

        output_shapes = ov::op::v1::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    }

    auto output_type = desc->output_data_types[0].value_or(input0_layout.data_type);
    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    return { layout{output_shapes[0], output_type, input0_layout.format} };
}

template std::vector<layout> space_to_batch_inst::calc_output_layouts<ov::PartialShape>(space_to_batch_node const& node, const kernel_impl_params& impl_param);

std::string space_to_batch_inst::to_string(space_to_batch_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite space_to_batch_info;
    space_to_batch_info.add("input id", input.id());

    node_info->add("space_to_batch_info", space_to_batch_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

space_to_batch_inst::typed_primitive_inst(network& network, space_to_batch_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
