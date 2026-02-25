// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adaptive_pooling_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "adaptive_avg_pool_shape_inference.hpp"
#include "adaptive_max_pool_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(adaptive_pooling)

layout adaptive_pooling_inst::calc_output_layout(adaptive_pooling_node const& node, kernel_impl_params const& impl_param) {
    const auto data_layout = impl_param.get_input_layout();
    const auto prim = impl_param.typed_desc<adaptive_pooling>();
    return {data_layout.data_type, data_layout.format, prim->output_size};
}

template<typename ShapeType>
std::vector<layout> adaptive_pooling_inst::calc_output_layouts(adaptive_pooling_node const& /*node*/, const kernel_impl_params& impl_param) {
    std::vector<layout> layouts;

    const auto prim = impl_param.typed_desc<adaptive_pooling>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto input1_layout = impl_param.get_input_layout(1);
    auto output_format = input0_layout.format;

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        input1_layout.get<ShapeType>()
    };
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::unordered_map<size_t, ov::Tensor> const_data;

    auto& memory_deps = impl_param.memory_deps;

    if (memory_deps.count(1)) {
        auto pooledVector_mem = memory_deps.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> pooledVector_lock(pooledVector_mem, impl_param.get_stream());

        auto pooledVector_tensor = make_tensor(pooledVector_mem->get_layout(), pooledVector_lock.data());

        const_data.emplace(1, pooledVector_tensor);
    }

    const auto tensor_accessor = ov::make_tensor_accessor(const_data);
    if (prim->mode == cldnn::adaptive_pooling_mode::max) {
        ov::op::v8::AdaptiveMaxPool op;

        output_shapes = ov::op::v8::shape_infer(&op, input_shapes, tensor_accessor);

        auto dt = prim->get_output_data_type(0).value_or(impl_param.get_input_layout(0).data_type);
        auto index_dt = prim->index_element_type;
        layouts.push_back(layout{output_shapes[0], dt, output_format});
        layouts.push_back(layout{output_shapes[1], index_dt, output_format});
    } else {
        ov::op::v8::AdaptiveAvgPool op;

        output_shapes = ov::op::v8::shape_infer(&op, input_shapes, tensor_accessor);

        auto dt = prim->get_output_data_type(0).value_or(impl_param.get_input_layout(0).data_type);
        layouts.push_back(layout{output_shapes[0], dt, output_format});
    }

    return layouts;
}

template std::vector<layout> adaptive_pooling_inst::calc_output_layouts<ov::PartialShape>(adaptive_pooling_node const& node,
                                                                                          const kernel_impl_params& impl_param);

std::string adaptive_pooling_inst::to_string(adaptive_pooling_node const& node) {
    const auto prim = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    const auto mode = prim->mode == adaptive_pooling_mode::max ? "max" : "average";
    info.add("mode", mode);
    info.add("output_size", prim->output_size);

    auto node_info = node.desc_to_json();
    node_info->add("adaptive_pooling_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

adaptive_pooling_inst::typed_primitive_inst(network& network, adaptive_pooling_node const& node) : parent(network, node) {}
}  // namespace cldnn
