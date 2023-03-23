// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"

namespace ov {
/** @brief Interface for data accessor. */
class OPENVINO_API ITensorAccessor {
public:
    /**
     * @brief Get tensor adapter at index
     *
     * @param port  Number of data port to get tensor adapter (operator input).
     * @return      Constant pointer to tensor adapter interface.
     */
    virtual Tensor operator()(size_t port) const = 0;

    virtual ~ITensorAccessor() = default;
};

/**
 * @brief Null tensor accessor.
 *
 * Return null pointer for any input port.
 */
struct OPENVINO_API NullTensorAccessor : public ITensorAccessor {
    Tensor operator()(size_t port) const override {
        const static auto null_tensor = Tensor();
        return null_tensor;
    }
};

/**
 * @brief Get null data adapter pointer for any index number.
 *
 * @param port  Port number to get data.
 * @return      Null pointer.
 */
inline auto null_tensor_accessor() -> const ITensorAccessor& {
    static const auto null_accessor = NullTensorAccessor();
    return null_accessor;
};

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
class OPENVINO_API TensorAccessor : public ITensorAccessor {
public:
    /**
     * @brief Construct a new Tensor Accessor object for tensors container.
     *
     * @param tensors  Pointer to container with tensors.
     */
    TensorAccessor(const TContainer* tensors) : m_tensors{tensors} {}

    /**
     * @brief Get tensor data adapter for given index.
     *
     * @param port  Port number to get data.
     *
     * @return Pointer to tensor adapter or nullptr if data not found at given index.
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
