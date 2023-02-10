// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_data_accessor.hpp"

namespace ov {
template <>
auto TensorAccessor<TensorVector>::operator()(size_t idx) const -> ITensorDataAdapter::UPtr {
    if (idx < m_tensors->size()) {
        return std::unique_ptr<TensorAdapter>(new TensorAdapter{&(*m_tensors)[idx]});
    } else {
        return {};
    }
}

template <>
auto TensorAccessor<HostTensorVector>::operator()(size_t idx) const -> ITensorDataAdapter::UPtr {
    if (idx < m_tensors->size()) {
        return std::unique_ptr<HostTensorAdapter>(new HostTensorAdapter{(*m_tensors)[idx].get()});
    } else {
        return {};
    }
}

template <>
auto TensorAccessor<std::map<size_t, HostTensorPtr>>::operator()(size_t idx) const -> ITensorDataAdapter::UPtr {
    const auto t_iter = m_tensors->find(idx);
    if (t_iter != m_tensors->cend()) {
        return std::unique_ptr<HostTensorAdapter>(new HostTensorAdapter{t_iter->second.get()});
    } else {
        return {};
    }
}
}  // namespace ov
