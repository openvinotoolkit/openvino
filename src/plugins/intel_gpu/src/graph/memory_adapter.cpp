// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_adapter.hpp"

namespace cldnn {

MemoryAdapter::MemoryAdapter(cldnn::memory::ptr ptr, const cldnn::stream& stream)
    : m_ptr{ptr},
      m_ptr_lock{ptr, stream} {}

ov::element::Type_t MemoryAdapter::get_element_type() const {
    return data_type_to_element_type(m_ptr->get_layout().data_type);
}

size_t MemoryAdapter::get_size() const {
    return ov::shape_size(m_ptr->get_layout().get_shape());
}

const void* MemoryAdapter::data() const {
    return m_ptr_lock.data();
}

}  // namespace cldnn
