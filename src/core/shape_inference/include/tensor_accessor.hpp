// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

/** \brief Function type to get tensors from memory container. */
using get_tensor_func_t = std::function<Tensor(size_t)>;

/**
 * \brief Functor implements get_tensor_func_t.
 *
 *  Is not owning TContainer and supports following tensor containers:
 *  - ov::TensorVector
 *  - ov::HostTensorVector
 *  - std::map<size_t, ov::HostTensorPtr>
 *
 * \tparam TContainer
 */
template <class TContainer>
struct TensorAccessor {
    TensorAccessor(const TContainer& tensors) : m_tensors{tensors} {}

    template <class C = TContainer, typename std::enable_if<std::is_same<C, TensorVector>::value>::type* = nullptr>
    Tensor operator()(size_t idx) {
        return idx < m_tensors.size() ? m_tensors[idx] : Tensor();
    }

    template <class C = TContainer, typename std::enable_if<std::is_same<C, HostTensorVector>::value>::type* = nullptr>
    Tensor operator()(size_t idx) {
        if (idx < m_tensors.size()) {
            return {m_tensors[idx]->get_element_type(), m_tensors[idx]->get_shape(), m_tensors[idx]->get_data_ptr()};
        } else {
            return {};
        }
    }

    template <class C = TContainer,
              typename std::enable_if<std::is_same<C, std::map<size_t, HostTensorPtr>>::value>::type* = nullptr>
    Tensor operator()(size_t idx) {
        const auto t_iter = m_tensors.find(idx);
        if (t_iter != m_tensors.cend()) {
            return {t_iter->second->get_element_type(), t_iter->second->get_shape(), t_iter->second->get_data_ptr()};
        } else {
            return {};
        }
    }

private:
    const TContainer& m_tensors;
};

/**
 * \brief Get default tensor for any index number.
 *
 * \param idx  Index to get tensor.
 * \return Default (null) ov::Tensor.
 */
inline auto null_tensor_accessor(size_t idx) -> ov::Tensor {
    return {};
};

/**
 * \brief Makes TensorAccessor for specific tensor container.
 *
 * \tparam TContainer Type of tensor containers \see TensorAccessor for supported types.
 *
 * \param c  Container of tensors.
 * \return TensorContainer for specific type.
 */
template <class TContainer>
auto make_tensor_accessor(const TContainer& c) -> TensorAccessor<TContainer> {
    return TensorAccessor<TContainer>(c);
}
}  // namespace ov
