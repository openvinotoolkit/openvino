// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"

namespace ov {
class Node;
class AttributeVisitor;
class Any;
using VariantTypeInfo = DiscreteTypeInfo;

class OPENVINO_API Variant {
public:
    static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static const ::ov::DiscreteTypeInfo type_info{"Variant", 0};
        return type_info;
    }
    virtual ~Variant();
    virtual const VariantTypeInfo& get_type_info() const {
        return get_type_info_static();
    }
    virtual bool is_copyable() const;
    virtual Any init(const std::shared_ptr<Node>& node);
    virtual Any merge(const ov::NodeVector& nodes);
    virtual std::string to_string() {
        return "";
    }
    virtual bool visit_attributes(AttributeVisitor&) {
        return false;
    }

    using type_info_t = DiscreteTypeInfo;
};

template <typename VT>
class VariantImpl : public Variant {
public:
    OPENVINO_RTTI(typeid(VT).name(), "0");
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

template <>
class VariantImpl<void> : public Variant {
public:
    using value_type = void;

    VariantImpl() = default;
};

extern template class OPENVINO_API VariantImpl<std::string>;
extern template class OPENVINO_API VariantImpl<int64_t>;
extern template class OPENVINO_API VariantImpl<bool>;

template <typename VT>
class VariantWrapper {};

template <>
class OPENVINO_API VariantWrapper<std::string> : public VariantImpl<std::string> {
public:
    OPENVINO_RTTI("VariantWrapper<std::string>");
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

template <>
class OPENVINO_API VariantWrapper<int64_t> : public VariantImpl<int64_t> {
public:
    OPENVINO_RTTI("VariantWrapper<int64_t>");
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

template <typename T>
inline std::shared_ptr<Variant> make_variant(const T& p) {
    return std::dynamic_pointer_cast<VariantImpl<T>>(std::make_shared<VariantWrapper<T>>(p));
}

template <size_t N>
inline std::shared_ptr<Variant> make_variant(const char (&s)[N]) {
    return std::dynamic_pointer_cast<VariantImpl<std::string>>(std::make_shared<VariantWrapper<std::string>>(s));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <size_t N>
inline std::shared_ptr<Variant> make_variant(const wchar_t (&s)[N]) {
    return std::dynamic_pointer_cast<VariantImpl<std::wstring>>(std::make_shared<VariantWrapper<std::wstring>>(s));
}
#endif
using VariantVector = std::vector<std::shared_ptr<Variant>>;
}  // namespace ov
