// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/memory_access.hpp"

#include <ngraph/runtime/host_tensor.hpp>

namespace ngraph {
namespace snippets {
namespace op {

MemoryAccess::MemoryAccess(const Output<Node>& x, const size_t count) : Op({x}), m_count(count) {
}

bool MemoryAccess::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("count", m_count);
    return true;
}

size_t MemoryAccess::get_count() const {
    return m_count;
}

void MemoryAccess::set_count(const size_t count) {
    m_count = count;
}

void MemoryAccess::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

} // namespace op
} // namespace snippets
} // namespace ngraph