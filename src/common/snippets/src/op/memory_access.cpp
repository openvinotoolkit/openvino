// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/memory_access.hpp"

#include <ngraph/runtime/host_tensor.hpp>

namespace ngraph {
namespace snippets {
namespace op {

MemoryAccess::MemoryAccess(const Output<Node>& x, const size_t count, const size_t offset) : Op({x}), m_count(count), m_offset(offset) {}

bool MemoryAccess::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("count", m_count);
    visitor.on_attribute("offset", m_offset);
    return true;
}

size_t MemoryAccess::get_count() const {
    return m_count;
}

size_t MemoryAccess::get_offset() const {
    return m_offset;
}

void MemoryAccess::set_count(const size_t count) {
    m_count = count;
}

void MemoryAccess::set_offset(const size_t offset) {
    m_offset = offset;
}

void MemoryAccess::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

} // namespace op
} // namespace snippets
} // namespace ngraph