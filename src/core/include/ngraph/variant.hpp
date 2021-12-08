// Copyright (C) 2018-2021 Intel Corporation
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
template <typename T>
using VariantImpl = ov::RuntimeAttributeImpl<T>;

template <typename T>
using VariantWrapper = ov::RuntimeAttributeWrapper<T>;

template <typename T>
inline std::shared_ptr<Variant> make_variant(const T& p) {
    return ov::make_runtime_attribute(p);
}

template <size_t N>
inline std::shared_ptr<Variant> make_variant(const char (&s)[N]) {
    return ov::make_runtime_attribute(s);
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <size_t N>
inline std::shared_ptr<Variant> make_variant(const wchar_t (&s)[N]) {
    return ov::make_runtime_attribute(s);
}
#endif

using ov::RTMap;
}  // namespace ngraph
