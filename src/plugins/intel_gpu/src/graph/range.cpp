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
    auto desc = impl_param.typed_desc<range>();
    auto output_data_type = desc->output_data_type.value_or(impl_param.get_input_layout().data_type);
    auto output_format = cldnn::format::bfyx;

    return layout(output_data_type, output_format, desc->output_shape);
}

template<typename ShapeType>
std::vector<layout> range_inst::calc_output_layouts(range_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<range>();
    auto input_layout = impl_param.get_input_layout(0);
    auto output_dt = desc->output_data_type.value_or(input_layout.data_type);
    auto output_format = cldnn::format::bfyx;

    auto& memory_deps = impl_param.memory_deps;
    if (!memory_deps.count(0) || !memory_deps.count(1) || !memory_deps.count(2)) {
        return { layout{ ShapeType::dynamic(1), output_dt, output_format } };
    }

    ov::op::v4::Range op;
    op.set_output_type(data_type_to_element_type(output_dt));
    std::vector<ShapeType> output_shapes = {ShapeType()};
    std::vector<ShapeType> input_shapes = {
        impl_param.get_input_layout(0).get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>(),
        impl_param.get_input_layout(2).get<ShapeType>()
    };

    auto start_mem = memory_deps.at(0);
    auto stop_mem = memory_deps.at(1);
    auto step_mem = memory_deps.at(2);

    cldnn::mem_lock<uint8_t, mem_lock_type::read> lock_start(start_mem, impl_param.prog.get_stream());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> lock_stop(stop_mem, impl_param.prog.get_stream());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> lock_step(step_mem, impl_param.prog.get_stream());

    auto tensor_start = make_host_tensor(start_mem->get_layout(), lock_start.data());
    auto tensor_stop = make_host_tensor(stop_mem->get_layout(), lock_stop.data());
    auto tensor_step = make_host_tensor(step_mem->get_layout(), lock_step.data());

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
        {0, tensor_start},
        {1, tensor_stop},
        {2, tensor_step},
    };

    ov::op::v4::shape_infer(&op, input_shapes, output_shapes, const_data);

    return { layout {output_shapes[0], output_dt, output_format} };
}

std::string range_inst::to_string(range_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite op_info;
    op_info.add("output_type", data_type_traits::name(desc->output_data_type.value()));

    node_info->add("range info", std::move(op_info));
    return lexical_cast(*node_info);
}

range_inst::typed_primitive_inst(network& network, range_node const& node) : typed_primitive_inst_base{network, node} {}

}  // namespace cldnn
