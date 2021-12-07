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

class OPENVINO_API RuntimeAttribute {
public:
    static const DiscreteTypeInfo& get_type_info_static() {
        static const ::ov::DiscreteTypeInfo type_info{"RuntimeAttribute", 0};
        return type_info;
    }
    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }
    using Ptr = std::shared_ptr<RuntimeAttribute>;
    using Base = std::tuple<::ov::RuntimeAttribute>;
    virtual ~RuntimeAttribute() = default;
    virtual bool is_copyable() const;
    virtual Any init(const std::shared_ptr<Node>& node) const;
    virtual Any merge(const ov::NodeVector& nodes) const;
    virtual std::string to_string() const;
    virtual bool visit_attributes(AttributeVisitor&);
    bool visit_attributes(AttributeVisitor& visitor) const {
        return const_cast<RuntimeAttribute*>(this)->visit_attributes(visitor);
    }
};
OPENVINO_API std::ostream& operator<<(std::ostream& os, const RuntimeAttribute& attrubute);

template <typename VT>
class RuntimeAttributeImpl : public RuntimeAttribute {
public:
    OPENVINO_RTTI(typeid(VT).name());
    using value_type = VT;

    RuntimeAttributeImpl() = default;

    RuntimeAttributeImpl(const value_type& value) : m_value(value) {}

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

extern template class OPENVINO_API RuntimeAttributeImpl<std::string>;
extern template class OPENVINO_API RuntimeAttributeImpl<int64_t>;
extern template class OPENVINO_API RuntimeAttributeImpl<bool>;

template <typename VT>
class RuntimeAttributeWrapper {};

template <>
class OPENVINO_API RuntimeAttributeWrapper<std::string> : public RuntimeAttributeImpl<std::string> {
public:
    OPENVINO_RTTI("RuntimeAttributeWrapper<std::string>");
    RuntimeAttributeWrapper(const value_type& value) : RuntimeAttributeImpl<value_type>(value) {}
    std::string to_string() const override {
        return m_value;
    }
};

template <>
class OPENVINO_API RuntimeAttributeWrapper<int64_t> : public RuntimeAttributeImpl<int64_t> {
public:
    OPENVINO_RTTI("RuntimeAttributeWrapper<int64_t>");
    RuntimeAttributeWrapper(const value_type& value) : RuntimeAttributeImpl<value_type>(value) {}
    std::string to_string() const override {
        return std::to_string(m_value);
    }
};

template <typename T>
inline std::shared_ptr<RuntimeAttribute> make_runtime_attribute(const T& p) {
    return std::static_pointer_cast<RuntimeAttribute>(std::make_shared<RuntimeAttributeWrapper<T>>(p));
}

template <size_t N>
inline std::shared_ptr<RuntimeAttribute> make_runtime_attribute(const char (&s)[N]) {
    return std::static_pointer_cast<RuntimeAttribute>(std::make_shared<RuntimeAttributeWrapper<std::string>>(s));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <size_t N>
inline std::shared_ptr<RuntimeAttribute> make_runtime_attribute(const wchar_t (&s)[N]) {
    return std::static_pointer_cast<RuntimeAttribute>(std::make_shared<RuntimeAttributeWrapper<std::wstring>>(s));
}
#endif

using RuntimeAttributeVector = std::vector<ov::Any>;
}  // namespace ov
