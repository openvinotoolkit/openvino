// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_data_accessor.hpp"

namespace ov {
template <>
const ITensorDataAdapter* TensorAccessor<TensorVector>::operator()(size_t port) const {
    if (port < m_tensors->size()) {
        m_adapter.m_ptr = &(*m_tensors)[port];
        return &m_adapter;
    } else {
        return nullptr;
    }
}

template <>
const ITensorDataAdapter* TensorAccessor<HostTensorVector>::operator()(size_t port) const {
    if (port < m_tensors->size()) {
        m_adapter.m_ptr = (*m_tensors)[port].get();
        return &m_adapter;
    } else {
        return nullptr;
    }
}

template <>
const ITensorDataAdapter* TensorAccessor<std::map<size_t, HostTensorPtr>>::operator()(size_t port) const {
    const auto t_iter = m_tensors->find(port);
    if (t_iter != m_tensors->cend()) {
        m_adapter.m_ptr = t_iter->second.get();
        return &m_adapter;
    } else {
        return nullptr;
    }
}
}  // namespace ov
