// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
/** @brief Interface for data accessor. */
class OPENVINO_API ITensorAccessor {
public:
    /**
     * @brief Get tensor at port.
     *
     * @param port  Number of data port (operator input) to get tensor.
     * @return      Tensor to data at port.
     */
    virtual Tensor operator()(size_t port) const = 0;

    virtual ~ITensorAccessor() = default;
};

/**
 * @brief Null tensor accessor.
 *
 * Return empty tensor for any input port.
 */
struct OPENVINO_API NullTensorAccessor : public ITensorAccessor {
    Tensor operator()(size_t port) const override {
        const static auto null_tensor = Tensor();
        return null_tensor;
    }
};

/**
 * @brief Get null tensor accessor which returns empty tensor for any index number.
 *
 * @param port  Port number to get data.
 * @return      Null tensor accessor.
 */
inline auto null_tensor_accessor() -> const ITensorAccessor& {
    static const auto null_accessor = NullTensorAccessor();
    return null_accessor;
};

/**
 * @brief Tensor data accessor functor.
 *
 * Creates the ov::Tensor found in tensors container.
 * This accessor not take ownership of tensors container.
 * Supports following containers:
 * - ov::TensorVector
 * - ov::HostTensorVector
 * - std::map<size_t, ov::HostTensorPtr>
 *
 * @tparam TContainer Type of tensor container.
 */
template <class TContainer>
class OPENVINO_API TensorAccessor : public ITensorAccessor {
public:
    /**
     * @brief Construct a new Tensor Accessor object for tensors container.
     *
     * @param tensors  Pointer to container with tensors.
     */
    TensorAccessor(const TContainer* tensors) : m_tensors{tensors} {}

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
OPENVINO_API Tensor TensorAccessor<TensorVector>::operator()(size_t port) const;

template <>
OPENVINO_API Tensor TensorAccessor<HostTensorVector>::operator()(size_t port) const;

template <>
OPENVINO_API Tensor TensorAccessor<std::map<size_t, HostTensorPtr>>::operator()(size_t port) const;

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
