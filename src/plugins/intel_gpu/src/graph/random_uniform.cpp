// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include "random_uniform_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(random_uniform)

random_uniform_inst::typed_primitive_inst(network& network, random_uniform_node const &node)
: parent(network, node) {
}

layout random_uniform_inst::calc_output_layout(random_uniform_node const &node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<random_uniform>();
    auto format = format::get_default_format(primitive->output_shape.size());

    return {primitive->output_shape, *primitive->output_data_types[0], format};
}

template<typename ShapeType>
std::vector<layout> random_uniform_inst::calc_output_layouts(random_uniform_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<random_uniform>();
    auto output_data_type = desc->output_data_types[0].value_or(impl_param.get_input_layout().data_type);

    std::vector<ShapeType> output_shapes;
    std::vector<ShapeType> input_shapes = { impl_param.get_input_layout(0).get_partial_shape(),
                                            impl_param.get_input_layout(1).get_partial_shape(),
                                            impl_param.get_input_layout(2).get_partial_shape() };

    auto& memory_deps = impl_param.memory_deps;
    std::unordered_map<size_t, ov::Tensor> const_data;

    auto run_shape_infer = [&]() {
        ov::op::v8::RandomUniform op;
        if (memory_deps.count(1) > 0 && memory_deps.count(2) > 0) {
            auto min_val = memory_deps.at(1);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> min_val_lock(min_val, impl_param.get_stream());
            const_data.emplace(1, make_tensor(min_val->get_layout(), min_val_lock.data()));

            auto max_val = memory_deps.at(2);
            cldnn::mem_lock<uint8_t, mem_lock_type::read> max_val_lock(max_val, impl_param.get_stream());
            const_data.emplace(2, make_tensor(max_val->get_layout(), max_val_lock.data()));

            return ov::op::v8::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
        } else {
            return ov::op::v8::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
        }
    };

    if (memory_deps.count(0) > 0) {
        auto output_shape = memory_deps.at(0);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> output_shape_lock(output_shape, impl_param.get_stream());
        const_data.emplace(0, make_tensor(output_shape->get_layout(), output_shape_lock.data()));

        output_shapes = run_shape_infer();
    } else {
        output_shapes = run_shape_infer();
    }

    return { layout{output_shapes[0], output_data_type, format::get_default_format(output_shapes[0].size())} };
}

template std::vector<layout> random_uniform_inst::calc_output_layouts<ov::PartialShape>(random_uniform_node const& node, const kernel_impl_params& impl_param);

std::string random_uniform_inst::to_string(random_uniform_node const &node) {
    auto node_info = node.desc_to_json();
    json_composite random_uniform_info;
    random_uniform_info.add("input id", node.input().id());
    random_uniform_info.add("min_value id", node.input(1).id());
    random_uniform_info.add("max_value  id", node.input(2).id());
    random_uniform_info.add("global_seed", node.get_primitive()->global_seed);
    random_uniform_info.add("op_seed", node.get_primitive()->op_seed);
    node_info->add("random uniform info", random_uniform_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

} // namespace cldnn
