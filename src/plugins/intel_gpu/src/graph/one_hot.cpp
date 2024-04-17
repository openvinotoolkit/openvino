// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot_inst.h"

#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <vector>

#include "one_hot_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(one_hot)

template<typename ShapeType>
std::vector<layout> one_hot_inst::calc_output_layouts(const one_hot_node& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<one_hot>();
    auto input_layout = impl_param.get_input_layout(0);
    auto dt = desc->output_data_types[0].value_or(input_layout.data_type);

    ov::op::v1::OneHot op;
    try {
        // set_axis also calls resolve_axis method which tries to get input0 partial shape
        // thus wrap this call with try/catch.
        // it's safe as shape_infer method calls normalize_axis internally
        op.set_axis(desc->one_hot_axis);
    } catch (...) {}

    std::vector<ShapeType> input_shapes = {
        input_layout.get_partial_shape(),
        ShapeType{},
        ShapeType{},
        ShapeType{}
    };

    int64_t depth = desc->depth;
    auto& memory_deps = impl_param.memory_deps;

    std::unordered_map<size_t, ov::Tensor> const_data = {};
    if (depth != 0) {
        auto depth_tensor = ov::Tensor(ov::element::i64, ov::Shape{1}, static_cast<void*>(&depth));
        const_data[1] = depth_tensor;
    } else if (memory_deps.count(1) > 0) {
        auto depth_mem = memory_deps.at(1);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> depth_lock(depth_mem, impl_param.get_stream());
        auto depth_ptr = depth_lock.data();

        // update depth_tensor if depth value comes from memory_deps instead of Constant node
        auto depth_tensor = make_tensor(depth_mem->get_layout(), depth_ptr);
        const_data[1] = depth_tensor;
    }

    std::vector<ShapeType> output_shapes =
        ov::op::v1::shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    return {{output_shapes[0], dt, format::get_default_format(output_shapes[0].size())}};
}

template std::vector<layout> one_hot_inst::calc_output_layouts<ov::PartialShape>(one_hot_node const& node, const kernel_impl_params& impl_param);

std::string one_hot_inst::to_string(one_hot_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    const auto& one_hot_axis = desc->one_hot_axis;
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite one_hot_info;
    one_hot_info.add("input id", input.id());
    one_hot_info.add("one-hot axis", one_hot_axis);

    node_info->add("one_hot info", one_hot_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

one_hot_inst::typed_primitive_inst(network& network, one_hot_node const& node) : parent(network, node) { }
}  // namespace cldnn
