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
#include "openvino/core/rtti.hpp"

namespace ov {

template <>
class FRONTEND_API VariantWrapper<std::istream*> : public VariantImpl<std::istream*> {
public:
    OPENVINO_RTTI("Variant::std::istream*");
    BWDCMP_RTTI_DECLARATION;

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

template <>
class FRONTEND_API VariantWrapper<std::istringstream*> : public VariantImpl<std::istringstream*> {
public:
    OPENVINO_RTTI("Variant::std::istringstream*");
    BWDCMP_RTTI_DECLARATION;

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
class FRONTEND_API VariantWrapper<std::wstring> : public VariantImpl<std::wstring> {
public:
    OPENVINO_RTTI("Variant::std::wstring");
    BWDCMP_RTTI_DECLARATION;

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};
#endif

using Weights = std::shared_ptr<ngraph::runtime::AlignedBuffer>;

template <>
class FRONTEND_API VariantWrapper<Weights>
    : public VariantImpl<Weights> {
public:
    OPENVINO_RTTI("Variant::Weights");
    BWDCMP_RTTI_DECLARATION;

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

using WeightsVariant = VariantWrapper<Weights>;

using Extensions = std::map<std::string, ngraph::OpSet>;

template <>
class FRONTEND_API VariantWrapper<Extensions>
    : public VariantImpl<Extensions> {
public:
    OPENVINO_RTTI("Variant::Extensions");
    BWDCMP_RTTI_DECLARATION;

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

using ExtensionsVariant = VariantWrapper<Extensions>;

}  // namespace ov