// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "openvino/core/except.hpp"

namespace ov {
namespace reference {
namespace bincount_detail {

template <typename TData>
size_t normalize_value(const TData value) {
    if constexpr (std::is_signed<TData>::value) {
        OPENVINO_ASSERT(value >= 0, "Bincount input data must be non-negative.");
    }

    using plain_t = typename std::remove_cv<TData>::type;
    using unsigned_t = typename std::make_unsigned<plain_t>::type;
    const auto unsigned_value = static_cast<unsigned_t>(value);
    const auto max_supported = static_cast<unsigned_t>(std::numeric_limits<int64_t>::max() - 1);
    OPENVINO_ASSERT(unsigned_value <= max_supported, "Bincount input value is too large.");

    return static_cast<size_t>(unsigned_value);
}

}  // namespace bincount_detail

/// \brief Reference implementation of the Bincount operation (unweighted).
///
/// \param data       Pointer to input integer data.
/// \param n          Number of elements in data.
/// \param minlength  Minimum output length (unused in kernel, but kept for API symmetry).
/// \param out        Output pointer (type int64_t). Must be a pre-allocated buffer of size out_size.
/// \param out_size   Size of the output buffer.
template <typename TData>
void bincount(const TData* data, size_t n, int64_t /*minlength*/, int64_t* out, size_t out_size) {
    for (size_t i = 0; i < out_size; ++i) {
        out[i] = 0;
    }
    for (size_t i = 0; i < n; ++i) {
        const auto val = bincount_detail::normalize_value(data[i]);
        if (val < out_size) {
            out[val]++;
        }
    }
}

/// \brief Reference implementation of the Bincount operation (weighted).
///
/// \param data       Pointer to input integer data.
/// \param weights    Pointer to weights data.
/// \param n          Number of elements in data (and weights).
/// \param minlength  Minimum output length (unused in kernel, kept for API symmetry).
/// \param out        Output pointer (type TWeight). Must be a pre-allocated buffer of size out_size.
/// \param out_size   Size of the output buffer.
template <typename TData, typename TWeight>
void bincount_weighted(const TData* data,
                       const TWeight* weights,
                       size_t n,
                       int64_t /*minlength*/,
                       TWeight* out,
                       size_t out_size) {
    for (size_t i = 0; i < out_size; ++i) {
        out[i] = TWeight{0};
    }
    for (size_t i = 0; i < n; ++i) {
        const auto val = bincount_detail::normalize_value(data[i]);
        if (val < out_size) {
            out[val] += weights[i];
        }
    }
}

/// \brief Compute the output size for bincount.
///
/// \param data      Pointer to input integer data.
/// \param n         Number of elements.
/// \param minlength Minimum output length.
/// \return          Output size = max(max(data) + 1, minlength)
template <typename TData>
size_t bincount_output_size(const TData* data, size_t n, int64_t minlength) {
    size_t max_val = 0;
    bool has_value = false;
    for (size_t i = 0; i < n; ++i) {
        const auto val = bincount_detail::normalize_value(data[i]);
        if (!has_value || val > max_val) {
            max_val = val;
            has_value = true;
        }
    }

    const size_t from_data = has_value ? max_val + 1 : 0;
    return std::max(from_data, static_cast<size_t>(minlength));
}

}  // namespace reference
}  // namespace ov
