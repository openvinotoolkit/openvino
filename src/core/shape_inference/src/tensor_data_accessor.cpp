// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_data_accessor.hpp"

#include "ngraph/runtime/host_tensor.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ov {
template <>
Tensor TensorAccessor<TensorVector>::operator()(size_t port) const {
    if (port < m_tensors->size()) {
        return (*m_tensors)[port];
    } else {
        return make_tensor_accessor()(port);
    }
}

template <>
Tensor TensorAccessor<HostTensorVector>::operator()(size_t port) const {
    if (port < m_tensors->size()) {
        auto ptr = (*m_tensors)[port];
        return {ptr->get_element_type(), ptr->get_shape(), ptr->get_data_ptr()};
    } else {
        return make_tensor_accessor()(port);
    }
}

template <>
Tensor TensorAccessor<std::unordered_map<size_t, Tensor>>::operator()(size_t port) const {
    const auto t_iter = m_tensors->find(port);
    if (t_iter != m_tensors->cend()) {
        return t_iter->second;
    } else {
        return make_tensor_accessor()(port);
    }
}

template <>
Tensor TensorAccessor<std::map<size_t, ngraph::HostTensorPtr>>::operator()(size_t port) const {
    const auto t_iter = m_tensors->find(port);
    if (t_iter != m_tensors->cend()) {
        auto ptr = t_iter->second.get();
        return {ptr->get_element_type(), ptr->get_shape(), ptr->get_data_ptr()};
    } else {
        return make_tensor_accessor()(port);
    }
}

template <>
Tensor TensorAccessor<void>::operator()(size_t) const {
    static const auto empty = Tensor();
    return empty;
}

auto make_tensor_accessor() -> const TensorAccessor<void>& {
    static constexpr auto empty_tensor_accessor = TensorAccessor<void>(nullptr);
    return empty_tensor_accessor;
}
}  // namespace ov
