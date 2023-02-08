// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

struct ITensorDataAdapter;

using ITensorDataAdapterPtr = std::unique_ptr<ITensorDataAdapter>;
/** \brief Function type to get tensors from memory container. */
using get_tensor_func_t = std::function<ITensorDataAdapterPtr(size_t)>;

struct ITensorDataAdapter {
    virtual ~ITensorDataAdapter() = default;

    virtual element::Type_t get_element_type() const = 0;
    virtual size_t get_size() const = 0;
    virtual const void* data() const = 0;
};

class TensorDataAdapter : public ITensorDataAdapter {
public:
    TensorDataAdapter(const Tensor& tensor) : m_tensor{tensor} {}

    element::Type_t get_element_type() const override {
        return m_tensor.get_element_type();
    }

    size_t get_size() const override {
        return m_tensor.get_size();
    };

    const void* data() const override {
        return m_tensor.data();
    }

private:
    const Tensor& m_tensor;
};

template <class TContainer>
class ContainerDataAdapter : public ITensorDataAdapter {
public:
    ContainerDataAdapter(const TContainer& c) : m_c{c} {}

    element::Type_t get_element_type() const override {
        return element::from<typename TContainer::value_type>();
    }

    size_t get_size() const override {
        return m_c.size();
    };

    const void* data() const override {
        return m_c.data();
    }

private:
    const TContainer& m_c;
};

class HostTensorDataAdapter : public ITensorDataAdapter {
public:
    HostTensorDataAdapter(const HostTensor& tensor) : m_tensor{tensor} {}

    element::Type_t get_element_type() const override {
        return m_tensor.get_element_type();
    }

    size_t get_size() const override {
        return m_tensor.get_element_count();
    }

    const void* data() const override {
        return m_tensor.get_data_ptr();
    }

private:
    const HostTensor& m_tensor;
};

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
    auto operator()(size_t idx) -> ITensorDataAdapterPtr {
        if (idx < m_tensors.size()) {
            return std::unique_ptr<TensorDataAdapter>(new TensorDataAdapter{m_tensors[idx]});
        } else {
            return {};
        }
    }

    template <class C = TContainer, typename std::enable_if<std::is_same<C, HostTensorVector>::value>::type* = nullptr>
    auto operator()(size_t idx) -> ITensorDataAdapterPtr {
        if (idx < m_tensors.size()) {
            return std::unique_ptr<HostTensorDataAdapter>(new HostTensorDataAdapter{*m_tensors[idx]});
        } else {
            return {};
        }
    }

    template <class C = TContainer,
              typename std::enable_if<std::is_same<C, std::map<size_t, HostTensorPtr>>::value>::type* = nullptr>
    auto operator()(size_t idx) -> ITensorDataAdapterPtr {
        const auto t_iter = m_tensors.find(idx);
        if (t_iter != m_tensors.cend()) {
            return std::unique_ptr<HostTensorDataAdapter>(new HostTensorDataAdapter{*t_iter->second});
        } else {
            return {};
        }
    }

private:
    const TContainer& m_tensors;
};

/**
 * \brief Get null data adapter pointer for any index number.
 *
 * \param idx  Index to get tensor.
 * \return nullptr.
 */
inline auto null_tensor_accessor(size_t idx) -> ITensorDataAdapterPtr {
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
