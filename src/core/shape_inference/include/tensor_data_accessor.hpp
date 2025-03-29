// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
/** @brief Interface for data accessor. */
class ITensorAccessor {
public:
    /**
     * @brief Get tensor at port.
     *
     * @param port  Number of data port (operator input) to get tensor.
     * @return      Tensor to data at port.
     */
    virtual Tensor operator()(size_t port) const = 0;

protected:
    ~ITensorAccessor() = default;
};

/**
 * @brief Tensor data accessor functor.
 *
 * Creates the ov::Tensor found in tensors container.
 * This accessor does not take ownership of tensors container.
 * Supports following containers:
 * - ov::TensorVector
 * - std::unordered_map<size_t, ov::Tensor>
 *
 * @tparam TContainer Type of tensor container.
 */
template <class TContainer>
class TensorAccessor final : public ITensorAccessor {
public:
    /**
     * @brief Construct a new Tensor Accessor object for tensors container.
     *
     * @param tensors  Pointer to container with tensors.
     */
    constexpr TensorAccessor(const TContainer* tensors) : m_tensors{tensors} {}

    /**
     * @brief Get tensor for given port number.
     *
     * @param port  Port number to get data.
     *
     * @return Tensor to data or empty tensor if data not found.
     */
    Tensor operator()(size_t port) const override;

private:
    const TContainer* m_tensors;  //!< Pointer to tensor container.
};

template <>
Tensor TensorAccessor<TensorVector>::operator()(size_t port) const;

template <>
Tensor TensorAccessor<std::unordered_map<size_t, Tensor>>::operator()(size_t port) const;

template <>
Tensor TensorAccessor<void>::operator()(size_t port) const;

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
constexpr auto make_tensor_accessor(const TContainer& c) -> TensorAccessor<TContainer> {
    return TensorAccessor<TContainer>(&c);
}

/**
 * @brief Makes empty TensorAccessor which return empty tensor for any port number.
 *
 * @return TensorAccessor to return empty tensor.
 */
auto make_tensor_accessor() -> const TensorAccessor<void>&;
}  // namespace ov
