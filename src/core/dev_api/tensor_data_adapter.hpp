// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

/**
 * @brief Tensor adapter interface to read object data like tensor.
 *
 * @todo Replace it by ITensor and/or TensorView when implemented.
 */
struct OPENVINO_API ITensorDataAdapter {
    using UPtr = std::unique_ptr<ITensorDataAdapter>;  //!< Unique smart pointer to data adapter.

    /**
     * @brief Get the element type.
     * @return element::Type_t
     */
    virtual element::Type_t get_element_type() const = 0;

    /**
     * @brief Get the size.
     * @return Size as number of elements.
     */
    virtual size_t get_size() const = 0;

    /**
     * @brief Pointer to data.
     * @return const void pointer to data.
     */
    virtual const void* data() const = 0;

    virtual ~ITensorDataAdapter() = default;
};

/**
 * @brief Data adapter for ov::Tensor.
 * @note The adapter is not owning tensor, user has to ensure that tensor outlives this adapter.
 */
class OPENVINO_API TensorAdapter : public ITensorDataAdapter {
public:
    /**
     * @brief Construct a new Tensor Adapter object.
     *
     * @param ptr  Pointer to tensor.
     */
    TensorAdapter(const Tensor* ptr);

    element::Type_t get_element_type() const override;
    size_t get_size() const override;
    const void* data() const override;

private:
    const Tensor* m_ptr;
};

/**
 * @brief Data adapter for ngraph::HostTensor.
 * @note The adapter is not owning tensor, user has to ensure that tensor outlives this adapter.
 */
class OPENVINO_API HostTensorAdapter : public ITensorDataAdapter {
public:
    /**
     * @brief Construct a new Host Tensor Adapter object.
     *
     * @param ptr  Pointer to HostTensor.
     */
    HostTensorAdapter(const ngraph::runtime::HostTensor* ptr);

    element::Type_t get_element_type() const override;
    size_t get_size() const override;
    const void* data() const override;

private:
    const ngraph::runtime::HostTensor* m_ptr;
};

/**
 * @brief Data adapter for STL container.
 * @note The adapter is not owning container, user has to ensure that tensor outlives this adapter.
 */
template <class TContainer>
class ContainerDataAdapter : public ITensorDataAdapter {
public:
    /**
     * @brief Construct a new Container Data Adapter object.
     *
     * @param ptr  Pointer to container.
     */
    ContainerDataAdapter(const TContainer* ptr) : m_c_ptr{ptr} {}

    element::Type_t get_element_type() const override {
        return element::from<typename TContainer::value_type>();
    }

    size_t get_size() const override {
        return m_c_ptr->size();
    };

    const void* data() const override {
        return m_c_ptr->data();
    }

private:
    const TContainer* m_c_ptr;
};

}  // namespace ov
