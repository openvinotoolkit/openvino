// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include <limits>
#include <algorithm>

namespace InferenceEngine {

template <typename OutT, typename InT>
inline typename std::enable_if<
        std::is_integral<OutT>::value && std::is_integral<InT>::value &&
        std::is_signed<InT>::value &&
        !std::is_same<OutT, InT>::value,
        OutT>::type saturated_cast(InT value) {
    if (std::numeric_limits<OutT>::max() > std::numeric_limits<InT>::max() &&
            std::numeric_limits<OutT>::min() < std::numeric_limits<InT>::min()) {
        return static_cast<OutT>(value);
    }

    const InT max = std::numeric_limits<OutT>::max() < std::numeric_limits<InT>::max() ? std::numeric_limits<OutT>::max() :
                                                                                         std::numeric_limits<InT>::max();
    const InT min = std::numeric_limits<OutT>::min() > std::numeric_limits<InT>::min() ? std::numeric_limits<OutT>::min() :
                                                                                         std::numeric_limits<InT>::min();

    return std::min(std::max(value, min), max);
}

template <typename OutT, typename InT>
inline typename std::enable_if<
        std::is_integral<OutT>::value && std::is_integral<InT>::value &&
        std::is_unsigned<InT>::value &&
        !std::is_same<OutT, InT>::value,
        OutT>::type saturated_cast(InT value) {
    if (std::numeric_limits<OutT>::max() > std::numeric_limits<InT>::max()) {
        return static_cast<OutT>(value);
    }

    const InT max = std::numeric_limits<OutT>::max() < std::numeric_limits<InT>::max() ? std::numeric_limits<OutT>::max() :
                                                                                         std::numeric_limits<InT>::max();

    return std::min(value, max);
}

template <typename OutT, typename InT>
inline typename std::enable_if<
        std::is_same<OutT, InT>::value,
        OutT>::type saturated_cast(InT value) {
    return value;
}

} // namespace InferenceEngine
