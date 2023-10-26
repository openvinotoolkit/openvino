// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/broadcastload.hpp"

#include "openvino/reference/broadcast.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace op {

BroadcastLoad::BroadcastLoad(const Output<Node>& x, ov::PartialShape shape, size_t offset)
    : MemoryAccess({x}, std::set<size_t>{0}, std::set<size_t>{}), output_shape(std::move(shape)) {
    set_input_port_descriptor({1, offset}, 0);
    constructor_validate_and_infer_types();
}

bool BroadcastLoad::visit_attributes(AttributeVisitor& visitor) {
    MemoryAccess::visit_attributes(visitor);
    return true;
}

std::shared_ptr<Node> BroadcastLoad::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BroadcastLoad);
    check_new_args_count(this, new_args);
    return std::make_shared<BroadcastLoad>(new_args.at(0), output_shape, get_offset());
}

void BroadcastLoad::validate_and_infer_types() {
    // BroadcastLoad has memory access port only on input
    const auto input_ma_ports = get_memory_access_input_ports();
    const auto output_ma_ports = get_memory_access_output_ports();
    OPENVINO_ASSERT(input_ma_ports.size() == 1 && is_memory_access_input_port(0), "BroadcastLoad node must have memory access input port");
    OPENVINO_ASSERT(output_ma_ports.size() == 0, "BroadcastLoad node mustn't have memory access output port");
    set_output_type(0, get_input_element_type(0), output_shape);
}

} // namespace op
} // namespace snippets
} // namespace ov
