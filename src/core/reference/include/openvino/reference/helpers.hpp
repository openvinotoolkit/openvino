// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace reference {
template <typename T>
struct widen {
    using type = T;
};

template <>
struct widen<float> {
    using type = double;
};

template <>
struct widen<double> {
    using type = long double;
};
}  // namespace reference
}  // namespace ov
