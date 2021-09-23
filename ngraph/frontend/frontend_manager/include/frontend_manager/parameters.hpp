// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "frontend_manager_defs.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/opsets/opset.hpp"
#include "openvino/core/variant.hpp"

namespace ov {

template <>
class FRONTEND_API VariantWrapper<std::istream*> : public VariantImpl<std::istream*> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::std::istream*", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

template <>
class FRONTEND_API VariantWrapper<std::istringstream*> : public VariantImpl<std::istringstream*> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::std::istringstream*", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
class FRONTEND_API VariantWrapper<std::wstring> : public VariantImpl<std::wstring> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::std::wstring", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};
#endif

template <>
class FRONTEND_API VariantWrapper<std::shared_ptr<ngraph::runtime::AlignedBuffer>>
    : public VariantImpl<std::shared_ptr<ngraph::runtime::AlignedBuffer>> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::Weights", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

using Weights = std::shared_ptr<ngraph::runtime::AlignedBuffer>;
using WeightsVariant = VariantWrapper<Weights>;

template <>
class FRONTEND_API VariantWrapper<std::map<std::string, ngraph::OpSet>>
    : public VariantImpl<std::map<std::string, ngraph::OpSet>> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::Extensions", 0};
    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

using Extensions = std::map<std::string, ngraph::OpSet>;
using ExtensionsVariant = VariantWrapper<Extensions>;

}  // namespace ov