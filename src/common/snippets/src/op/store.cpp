// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/store.hpp"

#include <ngraph/runtime/host_tensor.hpp>

namespace ngraph {
namespace snippets {
namespace op {

snippets::op::Store::Store(const Output<Node>& x, const size_t count, const size_t offset) : MemoryAccess({x}) {
    m_output_ports.resize(get_output_size());
    set_output_port_descriptor({count, offset}, 0);
    constructor_validate_and_infer_types();
}

void snippets::op::Store::validate_and_infer_types() {
    // Store has memory access port only on output
    OPENVINO_ASSERT(get_input_port_count() == 0, "Store node mustn't have memory access input port");
    OPENVINO_ASSERT(get_output_port_count() == 1, "Store node must have memory access output port");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> snippets::op::Store::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Store_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Store>(new_args.at(0), get_count(), get_offset());
}

void Store::set_input_port_descriptor(const MemoryAccess::PortDescriptor& desc, const size_t i) {
    throw ov::Exception("Store node doesn't have memory access input port");
}

const MemoryAccess::PortDescriptor& Store::get_input_port_descriptor(const size_t i) const {
    throw ov::Exception("Store node doesn't have memory access input port");
}

} // namespace op
} // namespace snippets
} // namespace ngraph
