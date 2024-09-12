// Copyright (C) 2018-2024 Intel Corporation
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
