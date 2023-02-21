// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_data_accessor.hpp"

namespace ov {
template <>
auto TensorAccessor<TensorVector>::operator()(size_t idx) const -> const ITensorDataAdapter* {
    if (idx < m_tensors->size()) {
        m_adapter.m_ptr = &(*m_tensors)[idx];
        return &m_adapter;
    } else {
        return nullptr;
    }
}

template <>
auto TensorAccessor<HostTensorVector>::operator()(size_t idx) const -> const ITensorDataAdapter* {
    if (idx < m_tensors->size()) {
        m_adapter.m_ptr = (*m_tensors)[idx].get();
        return &m_adapter;
    } else {
        return nullptr;
    }
}

template <>
auto TensorAccessor<std::map<size_t, HostTensorPtr>>::operator()(size_t idx) const -> const ITensorDataAdapter* {
    const auto t_iter = m_tensors->find(idx);
    if (t_iter != m_tensors->cend()) {
        m_adapter.m_ptr = t_iter->second.get();
        return &m_adapter;
    } else {
        return nullptr;
    }
}
}  // namespace ov
