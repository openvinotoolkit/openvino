// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/core/any.hpp"  // used for ov::RTMap
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
class Node;
}
namespace ngraph {
using ov::Node;
using VariantTypeInfo = ov::DiscreteTypeInfo;

using Variant = ov::RuntimeAttribute;

template <typename VT>
class OPENVINO_DEPRECATED("Please use ov::Any to store VT directly") VariantImpl : public Variant {
public:
    OPENVINO_RTTI(typeid(VT).name());
    using value_type = VT;

    VariantImpl() = default;

    VariantImpl(const value_type& value) : m_value(value) {}

    const value_type& get() const {
        return m_value;
    }
    value_type& get() {
        return m_value;
    }
    void set(const value_type& value) {
        m_value = value;
    }

protected:
    value_type m_value;
};

template <typename VT>
class OPENVINO_DEPRECATED("Please use ov::Any to store VT directly") VariantWrapper {};

OPENVINO_SUPPRESS_DEPRECATED_START
template <>
class OPENVINO_API VariantWrapper<std::string> : public VariantImpl<std::string> {
public:
    OPENVINO_RTTI("VariantWrapper<std::string>");
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
    std::string to_string() const override {
        return m_value;
    }
};

template <>
class OPENVINO_API VariantWrapper<int64_t> : public VariantImpl<int64_t> {
public:
    OPENVINO_RTTI("VariantWrapper<int64_t>");
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
    std::string to_string() const override {
        return std::to_string(m_value);
    }
};

template <typename T>
inline std::shared_ptr<Variant> make_variant(const T& p) {
    return std::static_pointer_cast<Variant>(std::make_shared<VariantWrapper<T>>(p));
}

template <size_t N>
inline std::shared_ptr<Variant> make_variant(const char (&s)[N]) {
    return std::static_pointer_cast<Variant>(std::make_shared<VariantWrapper<std::string>>(s));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <size_t N>
inline std::shared_ptr<Variant> make_variant(const wchar_t (&s)[N]) {
    return std::static_pointer_cast<Variant>(std::make_shared<VariantWrapper<std::wstring>>(s));
}
#endif

using ov::RTMap;

OPENVINO_SUPPRESS_DEPRECATED_END
}  // namespace ngraph
