// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/core/variant.hpp"

namespace ngraph {
using ov::VariantTypeInfo;

using ov::Variant;
using ov::VariantImpl;
using ov::VariantWrapper;

template <typename T>
inline std::shared_ptr<Variant> make_variant(const T& p) {
    return ov::make_variant(p);
}
template <size_t N>
inline std::shared_ptr<Variant> make_variant(const char (&s)[N]) {
    return ov::make_variant(s);
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <size_t N>
inline std::shared_ptr<Variant> make_variant(const wchar_t (&s)[N]) {
    return ov::make_variant(s);
}
#endif

using ov::RTMap;
}  // namespace ngraph
