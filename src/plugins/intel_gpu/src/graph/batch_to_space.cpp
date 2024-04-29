// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <vector>

#include "batch_to_space_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(batch_to_space)

layout batch_to_space_inst::calc_output_layout(batch_to_space_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<batch_to_space>();

    auto input_layout = impl_param.get_input_layout();
    auto input_format = input_layout.format;

    auto output_type = desc->output_data_types[0].value_or(input_layout.data_type);

    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    const size_t spatial_num = format::spatial_num(input_format);

    const auto& block_shape = desc->block_shape;
    const auto& crops_begin = desc->crops_begin;
    const auto& crops_end = desc->crops_end;

    if (block_shape.batch[0] != 1)
        CLDNN_ERROR_MESSAGE(desc->id,
            "block_shape[0] is expected to be 1. Actual block_shape[0] is " +
            std::to_string(block_shape.batch[0]));

    if (crops_begin.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "crops_begin[0] is expected to be 0. Actual crops_begin[0] is " +
            std::to_string(crops_begin.batch[0]));

    if (crops_end.batch[0] != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "crops_end[0] is expected to be 0. Actual crops_end[0] is " +
            std::to_string(crops_end.batch[0]));

    size_t block_sizes_multiplied = block_shape.feature[0];
    for (size_t i = 0; i < spatial_num; ++i)
        block_sizes_multiplied *= block_shape.spatial[i];

    if (input_layout.batch() % block_sizes_multiplied != 0)
        CLDNN_ERROR_MESSAGE(desc->id,
            "The batch of the input tensor must be divisible by multiplied block sizes = " +
            std::to_string(block_sizes_multiplied));

    if (crops_begin.feature[0] + crops_end.feature[0] >= block_shape.feature[0] * input_layout.feature())
            CLDNN_ERROR_MESSAGE(desc->id,
                "Output dimensions must be positive");

    for (size_t i = 0; i < spatial_num; ++i)
        if (crops_begin.spatial[i] + crops_end.spatial[i] >= block_shape.spatial[i] * input_layout.spatial(i))
            CLDNN_ERROR_MESSAGE(desc->id,
                "Output dimensions must be positive");

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
std::vector<layout> batch_to_space_inst::calc_output_layouts(batch_to_space_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<batch_to_space>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto input0_shape = input0_layout.get<ShapeType>();
    auto input0_size = input0_shape.size();
    auto input0_format = input0_layout.format;

    auto& constant_mem = impl_param.memory_deps;
    auto block_data = desc->block_shape;
    auto begin_data = desc->crops_begin;
    auto end_data = desc->crops_end;

    auto output_type = desc->output_data_types[0].value_or(input0_layout.data_type);
    if (impl_param.has_fused_primitives())
        output_type = impl_param.get_output_element_type();

    if (desc->shape_constant == 0 && (!constant_mem.count(1) || !constant_mem.count(2) || !constant_mem.count(3))) {
        auto out_shape = ov::PartialShape::dynamic(input0_size);
        return { layout{out_shape, output_type, input0_format } };
    }

    ShapeType block_shape = desc->shape_constant == 0 ? impl_param.get_input_layout(1).get<ShapeType>() : ov::Shape{ input0_size };
    ShapeType begin_shape = desc->shape_constant == 0 ? impl_param.get_input_layout(2).get<ShapeType>() : ov::Shape{ input0_size };
    ShapeType end_shape = desc->shape_constant == 0 ? impl_param.get_input_layout(3).get<ShapeType>() : ov::Shape{ input0_size };

    ov::op::v1::BatchToSpace op;
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

    return { layout{output_shapes[0], output_type, input0_format} };
}

template std::vector<layout> batch_to_space_inst::calc_output_layouts<ov::PartialShape>(batch_to_space_node const& node, const kernel_impl_params& impl_param);

std::string batch_to_space_inst::to_string(batch_to_space_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite batch_to_space_info;
    batch_to_space_info.add("input id", input.id());

    node_info->add("batch_to_space_info", batch_to_space_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

batch_to_space_inst::typed_primitive_inst(network& network, batch_to_space_node const& node)
    : parent(network, node) {}

}  // namespace cldnn
