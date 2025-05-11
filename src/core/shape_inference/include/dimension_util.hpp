// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "openvino/core/dimension.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace util {
namespace dim {

template <class TDim, typename std::enable_if<std::is_arithmetic<TDim>::value>::type* = nullptr>
constexpr bool is_static(const TDim) {
    return true;
}

template <class TDim, typename std::enable_if<!std::is_arithmetic<TDim>::value>::type* = nullptr>
constexpr bool is_static(const TDim& d) {
    return d.is_static();
}

template <class TDim>
constexpr typename std::enable_if<std::is_arithmetic<TDim>::value, TDim>::type get_length(const TDim& d) {
    return d;
}

template <class TDim>
constexpr typename std::enable_if<!std::is_arithmetic<TDim>::value, typename TDim::value_type>::type get_length(
    const TDim& d) {
    return d.get_length();
}

constexpr int64_t inf_bound = -1;  //!< Infinite bound value for dimension.

/**
 * @brief Checks if dimension length is infinite bound (undefined).
 *
 * @tparam T    Type of dimension length.
 * @param dim   Dimension length value.
 * @return True if dimension length has infinite bound, otherwise false.
 */
template <class T>
constexpr bool is_inf_bound(const T dim) {
    return dim == static_cast<T>(inf_bound);
}

/**
 * @brief Convert static dimension length to ov::Dimension::value_type.
 *
 * As static dimension length type is size_t (bit-length depends on architecture) the maximum value (undefined)
 * is convert to ov::Dimension infinite bound.
 *
 * @tparam T   Static dimension type (size_t)
 * @tparam U   ov::Dimension::value_type
 * @param dim  Dimension length to convert.
 * @return Converted input value to ov::Dimension::value_type.
 */
template <class T, class U = typename Dimension::value_type>
constexpr typename std::enable_if<std::is_same<T, size_t>::value, U>::type value_convert(const T dim) {
    return is_inf_bound(dim) ? inf_bound : static_cast<U>(dim);
}

/**
 * @brief Conversion of dimension when input type is same as ov::Dimension::value_type.
 *
 * Return value as it is.
 *
 * @tparam T   Dimension type same as ov::Dimension::value_type.
 * @tparam U   Dimension::value_type.
 * @param dim  Dimension length to convert.
 * @return Same value as input.
 */
template <class T, class U = typename Dimension::value_type>
constexpr typename std::enable_if<std::is_same<T, U>::value, U>::type value_convert(const T dim) {
    return dim;
}

/**
 * @brief Calculate dilated dimension value.
 *
 * @param dim       Dimension size value.
 * @param dilation  Dilation value
 * @return          Dilated dimension value.
 */
template <class T>
constexpr auto dilated(const T dim, const T dilation) -> T {
    return (dim < 1) ? inf_bound : dilation * (dim - 1) + 1;
}

/**
 * @brief Calculate dilated dimension.
 *
 * @tparam TDim     Dimension type.
 * @param dim       Dimension.
 * @param dilation  Dilation value.
 * @return Return dimension after dilation.
 */
template <class TDim>
constexpr auto dilated(const TDim& dim, const typename TDim::value_type dilation) -> TDim {
    return (dim - 1) * dilation + 1;
}

/**
 * @brief Calculate padded dimension size as dim size + padding size
 *
 * @tparam TDim    Dimension type as dimension class value type or any arithmetic value.
 * @param dim      Dimension size value.
 * @param pad_num  Number of padding to add.
 * @return         Padded dimension value or infinite bound.
 */
template <class TDim>
constexpr typename std::enable_if<std::is_arithmetic<TDim>::value, TDim>::type padded(const TDim dim,
                                                                                      const int64_t pad_num) {
    return (is_inf_bound(dim) || (dim + pad_num < 0)) ? inf_bound : dim + pad_num;
}

/**
 * @brief Calculate padded dimension size as dim + padding size
 *
 * @note the Dimension + operator cannot be used if padding is '-1' which result add dynamic dimension.
 *
 * @tparam TDim    Dimension type as dimension class.
 * @param dim      Dimension.
 * @param pad_num  Number padding to add.
 * @return         Padded dimension.
 */
template <class TDim>
typename std::enable_if<std::is_class<TDim>::value, TDim>::type padded(const TDim& dim, const int64_t pad_num) {
    if (pad_num != 0) {
        auto ub = padded(dim.get_max_length(), pad_num);
        if (dim.is_static()) {
            return {ub};
        } else {
            return {padded(dim.get_min_length(), pad_num), ub};
        }
    } else {
        return dim;
    }
}

/**
 * @brief Calculate dimension padding required by filter/kernel properties.
 *
 * Provides pair of padding values as left padding is total value of required padding divided by 2 and right as
 * total required padding minus left padding.
 *
 * @param dim          input dimension to calculate its padding.
 * @param filter_size  Kernel size for input dimension.
 * @param dilation     Kernel dilation.
 * @param stride       Kernel stride.
 * @return Pair of left, right padding values for input dimension.
 */
template <
    class TDim,
    class T = typename std::conditional<std::is_arithmetic<TDim>::value, size_t, typename Dimension::value_type>::type>
inline std::pair<T, T> padding(const TDim& dim, const int64_t kernel_size, const int64_t dilation, int64_t stride) {
    if (dim::is_static(dim)) {
        const auto dim_size = static_cast<int64_t>(dim::get_length(dim));
        const auto dilated_kernel = dilated(kernel_size, dilation);
        const int64_t tmp = (dim_size + stride - 1) / stride;

        const auto padding = std::max<int64_t>(0, (tmp - 1) * stride + dilated_kernel - dim_size);
        const auto left_padding = padding / 2;
        return {left_padding, padding - left_padding};
    } else {
        // If input dimension is infinite or interval the padding will be set to 0
        // as operator cannot store paddings for both bounds.
        return {0, 0};
    }
}

/**
 * @brief Divide dimension using ceil rounding.
 *
 * @tparam TDim    Dimension type.
 * @tparam T       Dimension length value type.
 *
 * @param dim      Input dimension.
 * @param divisor  Dimension division.
 * @return Divided dimension with bounds round up.
 */
template <class TDim>
auto ceil_div(const TDim& dim, const typename TDim::value_type divisor) -> TDim {
    using T = decltype(divisor);
    if (dim.is_static()) {
        return {util::ceil_div<T>(dim.get_length(), divisor)};
    } else if (dim.get_max_length() == static_cast<T>(dim::inf_bound)) {
        return {util::ceil_div<T>(dim.get_min_length(), divisor), dim.get_max_length()};
    } else {
        return {util::ceil_div<T>(dim.get_min_length(), divisor), util::ceil_div<T>(dim.get_max_length(), divisor)};
    }
}

/**
 * @brief Divide dimension using floor rounding.
 *
 * @tparam TDim    Dimension type.
 * @tparam T       Dimension length value type.
 *
 * @param dim      Input dimension.
 * @param divisor  Dimension division.
 * @return Divided dimension with bound round down.
 */
template <class TDim>
auto floor_div(const TDim& dim, const typename TDim::value_type divisor) -> TDim {
    using T = decltype(divisor);
    if (dim.is_static()) {
        return {dim.get_length() / divisor};
    } else if (dim.get_max_length() == static_cast<T>(dim::inf_bound)) {
        return {dim.get_min_length() / divisor, dim.get_max_length()};
    } else {
        return {dim.get_min_length() / divisor, dim.get_max_length() / divisor};
    }
}

/**
 * @brief Check if dimension is empty.
 *
 * For static dimension the empty dimension is equal to zero dimension.
 *
 * @tparam TDim  Dimension type.
 * @param d      Dimension for check.
 * @return true if dimension is empty otherwise false.
 */
template <class TDim,
          typename std::enable_if<!std::is_same<Dimension, typename std::decay<TDim>::type>::value>::type* = nullptr>
bool is_empty(TDim&& d) {
    return d == typename std::decay<TDim>::type{};
}

/**
 * @brief Check if dimension is empty.
 *
 * For iv::Dimension the empty means that has no dimension at all.
 *
 * @tparam TDim  Dimension type.
 * @param d      Dimension for check.
 * @return true if dimension is empty otherwise false.
 */
template <class TDim,
          typename std::enable_if<std::is_same<Dimension, typename std::decay<TDim>::type>::value>::type* = nullptr>
bool is_empty(TDim&& d) {
    return d.get_interval().empty();
}

/**
 * @brief Check if dimension is evenly divisible.
 *
 * @tparam TDim     Dimension type.
 * @param quotient  Dimension to check.
 * @param dividend  Dividend to check.
 * @return true if dimension is divisible other wise false.
 */
template <class TDim>
bool is_divisible(const TDim& quotient, const typename TDim::value_type dividend) {
    return !is_empty(quotient / dividend);
}

/**
 * @brief Scale dimension size by floating point value.
 *
 * @tparam TDim  Dimension type.
 * @param d      Dimension to scale.
 * @param scale  Scale value for dimension.
 */
template <class TDim>
void scale(TDim& d, float scale) {
    using T = typename TDim::value_type;
    static constexpr float epsilon = 1.0e-6f;
    if (scale != 1.0f) {
        scale += epsilon;

        auto ub = d.get_max_length();
        ub = is_inf_bound(ub) ? static_cast<T>(inf_bound) : static_cast<T>(static_cast<float>(ub) * scale);

        if (d.is_static()) {
            d = TDim(ub);
        } else {
            d = TDim(static_cast<T>(static_cast<float>(d.get_min_length()) * scale), ub);
        }
    }
}
}  // namespace dim
}  // namespace util
}  // namespace ov
