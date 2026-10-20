// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include <string>
#include <utility>
#include <vector>

#include "json_object.h"
#include "primitive_type_base.h"
#include "selective_ssm_inst.h"
#include "selective_ssm_shape_inference.hpp"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(selective_ssm)

layout selective_ssm_inst::calc_output_layout(const selective_ssm_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> selective_ssm_inst::calc_output_layouts(const selective_ssm_node& node, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<selective_ssm>();
    const auto& all_inputs = node.get_input_layouts();
    OPENVINO_ASSERT(all_inputs.size() == 6, "selective_ssm must have 6 inputs");

    ov::op::internal::SelectiveSSM op;
    std::vector<ShapeType> input_shapes;
    input_shapes.reserve(all_inputs.size());
    for (size_t i = 0; i < all_inputs.size(); i++) {
        input_shapes.push_back(impl_param.get_input_layout(i).get<ShapeType>());
    }
    const auto output_shapes = ov::op::internal::shape_infer(&op, input_shapes);

    const auto x_layout = impl_param.get_input_layout(3);
    const auto state_layout = impl_param.get_input_layout(5);
    std::vector<layout> output_layouts;
    output_layouts.emplace_back(output_shapes[0], x_layout.data_type, x_layout.format);
    if (desc->output_size() == 2) {
        output_layouts.emplace_back(output_shapes[1], state_layout.data_type, state_layout.format);
    }
    return output_layouts;
}

template std::vector<layout> selective_ssm_inst::calc_output_layouts<ov::PartialShape>(const selective_ssm_node& node, const kernel_impl_params& impl_param);

std::string selective_ssm_inst::to_string(const selective_ssm_node& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    json_composite ssm_info;
    ssm_info.add("A", node.input(0).id());
    ssm_info.add("dt", node.input(1).id());
    ssm_info.add("B", node.input(2).id());
    ssm_info.add("x", node.input(3).id());
    ssm_info.add("C", node.input(4).id());
    ssm_info.add("recurrent_state", node.input(5).id());
    node_info->add("selective_ssm_info", ssm_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void selective_ssm_inst::update_shape() {
    parent::update_shape();
    update_empty_sequence_output();
}

void selective_ssm_inst::on_execute() {
    update_empty_sequence_output();
}

void selective_ssm_inst::update_empty_sequence_output() {
    OPENVINO_ASSERT(_outputs.size() == 2, "selective_ssm must have 2 outputs");
    const auto& sequence_output_layout = get_output_layout(0);
    const bool empty_sequence = sequence_output_layout.is_static() && sequence_output_layout.count() == 0;

    if (!empty_sequence) {
        if (_state_output_aliased) {
            _outputs[1] = std::move(_state_output_memory);
            _max_output_layout_count[1] = _state_output_max_layout_count;
            _state_output_max_layout_count = 0;
            _state_output_aliased = false;
            set_flag(ExecutionFlags::MEMORY_CHANGED);
        }
        return;
    }

    build_deps();
    const auto state_input = input_memory_ptr(5);
    OPENVINO_ASSERT(state_input != nullptr, "selective_ssm recurrent state input is not allocated");

    // Dynamic zero-sized outputs are not allocated by the generic path. Keep a zero-sized view backed
    // by a dummy allocation so optimized users of output 1 can observe that all outputs are available.
    if (!_outputs[0]) {
        auto dummy = _network.get_engine().allocate_memory(layout{{1}, data_types::u8, format::bfyx});
        _outputs[0] = _network.get_engine().reinterpret_buffer(*dummy, sequence_output_layout);
        _max_output_layout_count[0] = 0;
        set_flag(ExecutionFlags::MEMORY_CHANGED);
    }

    if (!_state_output_aliased) {
        _state_output_memory = _outputs[1];
        _state_output_max_layout_count = _max_output_layout_count[1];
        _state_output_aliased = true;
    }

    if (!_outputs[1] || !_network.get_engine().is_the_same_buffer(*_outputs[1], *state_input)) {
        _outputs[1] = state_input;
        _max_output_layout_count[1] = state_input->get_layout().get_linear_size();
        set_flag(ExecutionFlags::MEMORY_CHANGED);
    }

    set_flag(ExecutionFlags::SKIP);
}

}  // namespace cldnn
