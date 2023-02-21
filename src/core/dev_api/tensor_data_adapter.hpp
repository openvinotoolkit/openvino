// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {

template <class>
class TensorAccessor;

/**
 * @brief Tensor adapter interface to read object data like tensor.
 *
 * @todo Replace it by ITensor and/or TensorView when implemented.
 */
struct OPENVINO_API ITensorDataAdapter {
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
    TensorAdapter() = default;

    element::Type_t get_element_type() const override;
    size_t get_size() const override;
    const void* data() const override;

private:
    friend class TensorAccessor<TensorVector>;
    const Tensor* m_ptr{};
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
    HostTensorAdapter(const HostTensor* ptr);
    HostTensorAdapter() = default;

    element::Type_t get_element_type() const override;
    size_t get_size() const override;
    const void* data() const override;

private:
    friend class TensorAccessor<HostTensorVector>;
    friend class TensorAccessor<std::map<size_t, HostTensorPtr>>;
    const HostTensor* m_ptr{};
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
    ContainerDataAdapter(const TContainer& ptr) : m_ptr{&ptr} {}

    element::Type_t get_element_type() const override {
        return element::from<typename TContainer::value_type>();
    }

    size_t get_size() const override {
        return m_ptr->size();
    };

    const void* data() const override {
        return m_ptr->data();
    }

private:
    const TContainer* m_ptr{};
};

/**
 * @brief Trait to get adapter type for specific type.
 *
 * @tparam T  Input type for which corresponding adapter type is provided.
 */
template <class T>
struct adapter_type_for {};

template <>
struct adapter_type_for<Tensor> {
    using value_type = TensorAdapter;
};

template <>
struct adapter_type_for<std::shared_ptr<ngraph::runtime::HostTensor>> {
    using value_type = HostTensorAdapter;
};

template <>
struct adapter_type_for<std::pair<const size_t, std::shared_ptr<ngraph::runtime::HostTensor>>> {
    using value_type = HostTensorAdapter;
};

/** @brief Helper to provide adapter type from T. */
template <class T>
using adapter_type_for_t = typename adapter_type_for<T>::value_type;

}  // namespace ov
