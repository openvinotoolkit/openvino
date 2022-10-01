// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "non_zero_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {

// -----------------------------------------------
// count_nonzero
// -----------------------------------------------
primitive_type_id count_nonzero::type_id() {
    static primitive_type_base<count_nonzero> instance;
    return &instance;
}

layout count_nonzero_inst::calc_output_layout(count_nonzero_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for count_nonzero_node!");
    return layout{cldnn::data_types::i32, cldnn::format::bfyx, tensor{1, 1, 1, 4}};
}

template<typename ShapeType>
std::vector<layout> count_nonzero_inst::calc_output_layouts(count_nonzero_node const& /*node*/, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_type) == false &&
           "Output data type forcing is not supported for count_nonzero_node!");
    if (impl_param.input_layouts[0].is_dynamic()) {
        return { layout{ov::PartialShape{4}, cldnn::data_types::i32, cldnn::format::bfyx} };
    } else {
        auto rank =  static_cast<ov::Dimension::value_type>(impl_param.input_layouts[0].get_shape().size());
        auto out_layout = layout{ov::PartialShape{rank + 1}, cldnn::data_types::i32, cldnn::format::bfyx};
        return {out_layout};
    }
}

std::string count_nonzero_inst::to_string(count_nonzero_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite count_nonzero_info;
    count_nonzero_info.add("input id", input.id());
    count_nonzero_info.add("output shape", tensor{1, 1, 1, 4});

    node_info->add("count_nonzero info", count_nonzero_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

count_nonzero_inst::typed_primitive_inst(network& network, count_nonzero_node const& node) : parent(network, node) {}

void count_nonzero_inst::on_execute() {
    output_memory().fill(_network.get_stream(), 0);
}

// -----------------------------------------------
// gather_nonzero
// -----------------------------------------------
primitive_type_id gather_nonzero::type_id() {
    static primitive_type_base<gather_nonzero> instance;
    return &instance;
}

layout gather_nonzero_inst::calc_output_layout(gather_nonzero_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for gather_nonzero_node!");
    if (impl_param.memory_deps.count(1)) {
        auto out_size = read_vector<int64_t>(impl_param.memory_deps.at(1), impl_param.prog.get_stream());
        ov::Shape output_shape(out_size.begin(), out_size.end());
        ov::PartialShape output_pshape(output_shape);
        return layout{output_pshape, cldnn::data_types::i32, cldnn::format::bfyx};
    } else {
        return layout{ov::PartialShape({ov::Dimension::dynamic(), ov::Dimension::dynamic()}), cldnn::data_types::i32, cldnn::format::bfyx};
    }
}

std::string gather_nonzero_inst::to_string(gather_nonzero_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite gather_nonzero_info;
    gather_nonzero_info.add("input id", input.id());
    if (node.is_valid_output_layout()) {
        gather_nonzero_info.add("output layout", node.get_output_layout().to_string());
    }

    node_info->add("gather_nonzero info", gather_nonzero_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gather_nonzero_inst::typed_primitive_inst(network& network, gather_nonzero_node const& node) : parent(network, node, false) {}

}  // namespace cldnn
