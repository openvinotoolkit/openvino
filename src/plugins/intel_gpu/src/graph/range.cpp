// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "range_shape_inference.hpp"

namespace cldnn {
namespace {
std::string lexical_cast(const json_base& j, int offset = 1) {
    std::stringstream os;
    j.dump(os, offset);
    return os.str();
}
}  // namespace

GPU_DEFINE_PRIMITIVE_TYPE_ID(range)

layout range_inst::calc_output_layout(range_node const& node, kernel_impl_params const& impl_param) {
    return impl_param.typed_desc<range>()->output_layout;
}

template<typename ShapeType>
std::vector<layout> range_inst::calc_output_layouts(range_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<range>();
    auto output_data_type = desc->output_data_types[0].value_or(impl_param.get_input_layout().data_type);

    ov::op::v4::Range op;
    op.set_output_type(output_data_type);
    std::vector<ShapeType> output_shapes = {ShapeType::dynamic(1)};
    std::vector<ShapeType> input_shapes = {ov::Shape(), ov::Shape(), ov::Shape()};

    std::unordered_map<size_t, ov::Tensor> const_data;
    auto& memory_deps = impl_param.memory_deps;

    if (memory_deps.count(0) > 0 && memory_deps.count(1) > 0 && memory_deps.count(2) > 0) {
        auto start_mem = memory_deps.at(0);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> start_mem_lock(start_mem, impl_param.get_stream());
        const_data.emplace(0, make_tensor(start_mem->get_layout(), start_mem_lock.data()));

        auto stop_mem = memory_deps.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> stop_mem_lock(stop_mem, impl_param.get_stream());
        const_data.emplace(1, make_tensor(stop_mem->get_layout(), stop_mem_lock.data()));

        auto step_mem = memory_deps.at(2);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> step_mem_lock(step_mem, impl_param.get_stream());
        const_data.emplace(2, make_tensor(step_mem->get_layout(), step_mem_lock.data()));

        output_shapes = shape_infer(&op, input_shapes, ov::make_tensor_accessor(const_data));
    }

    return {layout({output_shapes[0], output_data_type, impl_param.get_output_layout().format})};
}

template std::vector<layout> range_inst::calc_output_layouts<ov::PartialShape>(range_node const& node, const kernel_impl_params& impl_param);

std::string range_inst::to_string(range_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite op_info;
    op_info.add("output_type", ov::element::Type(desc->output_layout.data_type));

    node_info->add("range info", std::move(op_info));
    return lexical_cast(*node_info);
}

range_inst::typed_primitive_inst(network& network, range_node const& node) : typed_primitive_inst_base{network, node} {}

}  // namespace cldnn
