// Copyright (C) 2018-2022 Intel Corporation
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

primitive_type_id range::type_id() {
    static primitive_type_base<range> instance;
    return &instance;
}

layout range_inst::calc_output_layout(range_node const& node, kernel_impl_params const& impl_param) {
    return impl_param.typed_desc<range>()->output_layout;
}

template<typename ShapeType>
std::vector<layout> range_inst::calc_output_layouts(range_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<range>();
    auto output_data_type = desc->output_data_type.value_or(impl_param.get_input_layout().data_type);

    ov::op::v4::Range op;
    op.set_output_type(data_type_to_element_type(output_data_type));
    std::vector<ShapeType> output_shapes = {ShapeType::dynamic(1)};
    std::vector<ShapeType> input_shapes = {ov::Shape(), ov::Shape(), ov::Shape()};

    std::map<size_t, ngraph::HostTensorPtr> const_data;
    auto& memory_deps = impl_param.memory_deps;

    if (memory_deps.count(0) > 0 && memory_deps.count(1) > 0 && memory_deps.count(2) > 0) {
        auto start_mem = memory_deps.at(0);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> start_mem_lock(start_mem, impl_param.prog.get_stream());
        const_data.emplace(0, make_host_tensor(start_mem->get_layout(), start_mem_lock.data()));

        auto stop_mem = memory_deps.at(1);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> stop_mem_lock(stop_mem, impl_param.prog.get_stream());
        const_data.emplace(1, make_host_tensor(stop_mem->get_layout(), stop_mem_lock.data()));

        auto step_mem = memory_deps.at(2);
        cldnn::mem_lock<uint8_t, mem_lock_type::read> step_mem_lock(step_mem, impl_param.prog.get_stream());
        const_data.emplace(2, make_host_tensor(step_mem->get_layout(), step_mem_lock.data()));

        shape_infer(&op, input_shapes, output_shapes, const_data);
    }

    return {layout({output_shapes[0], output_data_type, impl_param.output_layout.format})};
}

template std::vector<layout> range_inst::calc_output_layouts<ov::PartialShape>(range_node const& node, const kernel_impl_params& impl_param);

std::string range_inst::to_string(range_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite op_info;
    op_info.add("output_type", data_type_traits::name(desc->output_layout.data_type));

    node_info->add("range info", std::move(op_info));
    return lexical_cast(*node_info);
}

range_inst::typed_primitive_inst(network& network, range_node const& node) : typed_primitive_inst_base{network, node} {}

}  // namespace cldnn
