// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_data_adapter.hpp"

namespace ov {
TensorAdapter::TensorAdapter(const Tensor* ptr) : m_ptr{ptr} {}

element::Type_t TensorAdapter::get_element_type() const {
    return m_ptr->get_element_type();
}

size_t TensorAdapter::get_size() const {
    return m_ptr->get_size();
};

const void* TensorAdapter::data() const {
    return m_ptr->data();
}

HostTensorAdapter::HostTensorAdapter(const HostTensor* ptr) : m_ptr{ptr} {}

element::Type_t HostTensorAdapter::get_element_type() const {
    return m_ptr->get_element_type();
}

size_t HostTensorAdapter::get_size() const {
    return m_ptr->get_element_count();
}

const void* HostTensorAdapter::data() const {
    return m_ptr->get_data_ptr();
}
}  // namespace ov
