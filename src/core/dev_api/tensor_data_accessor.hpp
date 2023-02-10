// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "tensor_data_adapter.hpp"

namespace ov {

/** @brief Function object type to get data from object container as tensor data adapter. */
using tensor_data_accessor_func_t = std::function<ITensorDataAdapter::UPtr(size_t)>;

/**
 * @brief Tensor data accessor functor.
 *
 * Creates the tensor data adapter if found in tensor container.
 * This accessor not take ownership of tensors container.
 * Supports following containers:
 * - ov::TensorVector
 * - ov::HostTensorVector
 * - std::map<size_t, ov::HostTensorPtr>
 *
 * @tparam TContainer Type of tensor container.
 */
template <class TContainer>
struct OPENVINO_API TensorAccessor {
    /**
     * @brief Construct a new Tensor Accessor object for tensors container.
     *
     * @param tensors  Pointer to container with tensors.
     */
    TensorAccessor(const TContainer* tensors) : m_tensors{tensors} {}

    /**
     * @brief Get tensor data adapter for given index.
     *
     * @param idx  Index in tensor container.
     *
     * @return Pointer to tensor adapter or nullptr if data not found at given index.
     */
    auto operator()(size_t idx) const -> ITensorDataAdapter::UPtr;

private:
    const TContainer* m_tensors;  //!< Pointer to tensor container.
};

template <>
auto TensorAccessor<TensorVector>::operator()(size_t idx) const -> ITensorDataAdapter::UPtr;

template <>
auto TensorAccessor<HostTensorVector>::operator()(size_t idx) const -> ITensorDataAdapter::UPtr;

template <>
auto TensorAccessor<std::map<size_t, HostTensorPtr>>::operator()(size_t idx) const -> ITensorDataAdapter::UPtr;

/**
 * @brief Get null data adapter pointer for any index number.
 *
 * @param idx  Index to get tensor.
 *
 * @return nullptr.
 */
inline auto null_tensor_accessor(size_t idx) -> ITensorDataAdapter::UPtr {
    return {};
};

/**
 * @brief Makes TensorAccessor for specific tensor container.
 *
 * @tparam TContainer Type of tensor containers @see TensorAccessor for supported types.
 *
 * @param c  Container of tensors.
 *
 * @return TensorContainer for specific type.
 */
template <class TContainer>
auto make_tensor_accessor(const TContainer& c) -> TensorAccessor<TContainer> {
    return TensorAccessor<TContainer>(&c);
}

}  // namespace ov
