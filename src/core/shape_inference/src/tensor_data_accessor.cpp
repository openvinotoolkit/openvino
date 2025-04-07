// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_data_accessor.hpp"
namespace ov {
template <>
Tensor TensorAccessor<TensorVector>::operator()(const size_t port) const {
    return (port < m_tensors->size()) ? (*m_tensors)[port] : Tensor{};
}

template <>
Tensor TensorAccessor<std::unordered_map<size_t, Tensor>>::operator()(const size_t port) const {
    const auto t_iter = m_tensors->find(port);
    return (t_iter != m_tensors->cend()) ? t_iter->second : Tensor{};
}

template <>
Tensor TensorAccessor<void>::operator()(const size_t) const {
    return {};
}

auto make_tensor_accessor() -> const TensorAccessor<void>& {
    static constexpr auto empty_tensor_accessor = TensorAccessor<void>(nullptr);
    return empty_tensor_accessor;
}
}  // namespace ov
