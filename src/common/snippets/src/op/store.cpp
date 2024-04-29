// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/store.hpp"


namespace ov {
namespace snippets {
namespace op {

snippets::op::Store::Store(const Output<Node>& x, const size_t count, const size_t offset)
    : MemoryAccess(std::set<size_t>{}, std::set<size_t>{0}), Op({x}) {
    set_output_port_descriptor({count, offset}, 0);
    constructor_validate_and_infer_types();
}

void snippets::op::Store::validate_and_infer_types() {
    // Store has memory access port only on output
    const auto input_ma_ports = get_memory_access_input_ports();
    const auto output_ma_ports = get_memory_access_output_ports();
    OPENVINO_ASSERT(input_ma_ports.size() == 0, "Store node mustn't have memory access input port");
    OPENVINO_ASSERT(output_ma_ports.size() == 1 && is_memory_access_output_port(0), "Store node must have memory access output port");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> snippets::op::Store::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Store_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Store>(new_args.at(0), get_count(), get_offset());
}
bool snippets::op::Store::visit_attributes(AttributeVisitor& visitor) {
    return MemoryAccess::visit_attributes(visitor);
}
} // namespace op
} // namespace snippets
} // namespace ov
