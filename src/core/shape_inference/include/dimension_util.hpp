// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "openvino/util/common_util.hpp"

namespace ov {
namespace util {
namespace dim {

constexpr auto inf_bound = -1;  //!< Infinite bound value for dimension.

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
 * @param dim      Dimension size value.
 * @param pad_num  Number of padded dimension.
 * @return         Padded dimension value or infinite bound.
 */
constexpr auto padded(const int64_t dim, const int64_t pad_num) -> int64_t {
    return ((dim == inf_bound) || (dim + pad_num < 0)) ? inf_bound : dim + pad_num;
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
template <class TDim, class T = typename TDim::value_type>
inline std::pair<T, T> padding(const TDim& dim, const int64_t kernel_size, const int64_t dilation, int64_t stride) {
    if (dim.is_static()) {
        const auto dim_size = static_cast<int64_t>(dim.get_length());
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
        return {dim};
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
        return {dim};
    } else {
        return {dim.get_min_length() / divisor, dim.get_max_length() / divisor};
    }
}

}  // namespace dim
}  // namespace util
}  // namespace ov
