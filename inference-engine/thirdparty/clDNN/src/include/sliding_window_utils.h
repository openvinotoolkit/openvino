// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <api/CPP/layout.hpp>
#include <api/CPP/tensor.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "meta_utils.h"

namespace cldnn {

/// @brief Sliding window output range computation mode.
enum class swor_mode {
    // Single modes:
    all,          ///< Range is computed in the way that each sliding window in range is fully contained inside
                  ///< (optionally upper-padded by offset) input data.
    exceed_once,  ///< Range is computed in the way that each except at most one sliding window in range is fully
                  ///< contained inside (optionally upper-padded by offset) input data. The last window may partially
                  ///< exceed (optionally upper-padded by offset) input data range.
    any,          ///< Range is computed in the way that each sliding window in range is fully or at least partially
                  ///< contained inside (optionally upper-padded by offset) input data.
    // Mixed modes:
    exceed_once_data,  ///< Range is computed in the way that each except at most one sliding window in range is fully
                       ///< contained inside (optionally upper-padded by offset) input data. The last window may
                       ///< partially exceed (non-upper-padded) input data range.
                       ///< This mode is effectievely minimum of combination of @c swor_mode::exceed_once mode
                       ///< and @c swor_mode::any mode (with always @c sym_offset = false).
    max                ///< Maximum of all single modes with all cominations of @c sym_offset.
};

/// @brief Calculates output range (size) for sliding window moving on input data range specified by @p input_size.
///
/// @param input_size Range/Size of input data (non-padded or treated as valid). Only spatial coordinates are
///                   considered.
/// @param size       Size of sliding window. Only spatial coordinates are considered.
/// @param offset     Offset/Padding of sliding window in input. Only spatial coordinates are considered. Padding/Offset
///                   is applied from both sides of input data: negative value extends/pads data, positive - crops it.
/// @param stride     Horizontal/Vertical stride of sliding in input data.
/// @param dilation   Horizontal/Vertical dilation of sliding window on input data.
/// @param sym_offset Treat offset as applied on input symmetrically (from both sides). If @c false, the @p offset
///                   is applied only from left/upper side.
/// @param degen_val  If values from calculation are in allowed range, but calculated output size is invalid,
///                   the @p degen_val is returned. Any non-positive value is considered degenerated and will be
///                   switched to value passed in this parameter.
/// @return Output range (size) of sliding window. Only spatial dimensions are valid (rest is 0).
template <swor_mode RangeMode = swor_mode::all>
tensor calc_sliding_window_output_range(const tensor& input_size,
                                        const tensor& size,
                                        const tensor& offset,
                                        const tensor& stride,
                                        const tensor& dilation = {1, 1, 1, 1},
                                        bool sym_offset = true,
                                        const tensor::value_type& degen_val = 0);

/// @brief Fall-back implementation.
template <swor_mode RangeMode>
tensor calc_sliding_window_output_range(const tensor&,
                                        const tensor&,
                                        const tensor&,
                                        const tensor&,
                                        const tensor&,
                                        bool,
                                        const tensor::value_type&) {
    static_assert(meta::always_false<meta::val_tuple<swor_mode, RangeMode>>::value,
                  "Sliding window output range calculation mode is not supported. Please implement specialization "
                  "for new swor_mode.");

    return tensor();
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::all>(const tensor& input_size,
                                                               const tensor& size,
                                                               const tensor& offset,
                                                               const tensor& stride,
                                                               const tensor& dilation,
                                                               bool sym_offset,
                                                               const tensor::value_type& degen_val) {
    if (input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0 || input_size.spatial[2] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (stride.spatial[0] <= 0 || stride.spatial[1] <= 0 || stride.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if (dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0 || dilation.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;
    tensor wnd_ext_size{0,
                        0,
                        (size.spatial[0] - 1) * dilation.spatial[0] + 1,
                        (size.spatial[1] - 1) * dilation.spatial[1] + 1,
                        (size.spatial[2] - 1) * dilation.spatial[2] + 1};

    // wes = (size - 1) * dilation + 1
    // lpos(i) = offset + i * stride + wes - 1,   for i = 0, 1, ...
    //
    // output_range = max {i | lpos(i) < input_size - offset} + 1,   if sym_offset is true
    // output_range = max {i | lpos(i) < input_size} + 1,            if sym_offset is false
    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[0] + wnd_ext_size.spatial[0] <= input_size.spatial[0]
            ? (input_size.spatial[0] - off_factor * offset.spatial[0] - wnd_ext_size.spatial[0]) / stride.spatial[0] + 1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[1] + wnd_ext_size.spatial[1] <= input_size.spatial[1]
            ? (input_size.spatial[1] - off_factor * offset.spatial[1] - wnd_ext_size.spatial[1]) / stride.spatial[1] + 1
            : degen_val);
    auto output_range_z = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[2] + wnd_ext_size.spatial[2] <= input_size.spatial[2]
            ? (input_size.spatial[2] - off_factor * offset.spatial[2] - wnd_ext_size.spatial[2]) / stride.spatial[2] + 1
            : degen_val);

    return {0, 0, output_range_x, output_range_y, output_range_z};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::exceed_once>(const tensor& input_size,
                                                                       const tensor& size,
                                                                       const tensor& offset,
                                                                       const tensor& stride,
                                                                       const tensor& dilation,
                                                                       bool sym_offset,
                                                                       const tensor::value_type& degen_val) {
    if (input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0 || input_size.spatial[2] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (stride.spatial[0] <= 0 || stride.spatial[1] <= 0 || stride.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if (dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0 || dilation.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;
    tensor wnd_ext_size{0,
                        0,
                        (size.spatial[0] - 1) * dilation.spatial[0] + 1,
                        (size.spatial[1] - 1) * dilation.spatial[1] + 1,
                        (size.spatial[2] - 1) * dilation.spatial[2] + 1};

    tensor extend = tensor::max(wnd_ext_size, stride);

    // wes = (size - 1) * dilation + 1
    // fpos(i) = offset + i * stride,             for i = 0, 1, ...
    // lpos(i) = offset + i * stride + wes - 1,   for i = 0, 1, ...
    //
    // output_range = max {i | lpos(i) < input_size - offset - 1 and fpos(i + 1) < input_size - offset} + 2,   if
    // sym_offset is true output_range = max {i | lpos(i) < input_size - 1          and fpos(i + 1) < input_size} + 2,
    // if sym_offset is false
    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[0] + extend.spatial[0] <= input_size.spatial[0] + stride.spatial[0] - 1
            ? (input_size.spatial[0] - off_factor * offset.spatial[0] - extend.spatial[0] + stride.spatial[0] - 1) /
                      stride.spatial[0] +
                  1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[1] + extend.spatial[1] <= input_size.spatial[1] + stride.spatial[1] - 1
            ? (input_size.spatial[1] - off_factor * offset.spatial[1] - extend.spatial[1] + stride.spatial[1] - 1) /
                      stride.spatial[1] +
                  1
            : degen_val);
    auto output_range_z = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[2] + extend.spatial[2] <= input_size.spatial[2] + stride.spatial[2] - 1
            ? (input_size.spatial[2] - off_factor * offset.spatial[2] - extend.spatial[2] + stride.spatial[2] - 1) /
                      stride.spatial[2] +
                  1
            : degen_val);

    return {0, 0, output_range_x, output_range_y, output_range_z};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::any>(const tensor& input_size,
                                                               const tensor& size,
                                                               const tensor& offset,
                                                               const tensor& stride,
                                                               const tensor& dilation,
                                                               bool sym_offset,
                                                               const tensor::value_type& degen_val) {
    if (input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0 || input_size.spatial[2] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (stride.spatial[0] <= 0 || stride.spatial[1] <= 0 || stride.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if (dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0 || dilation.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;

    // fpos(i) = offset + i * stride,             for i = 0, 1, ...
    //
    // output_range = max {i | fpos(i) < input_size - offset} + 1,   if sym_offset is true
    // output_range = max {i | fpos(i) < input_size} + 1,            if sym_offset is false
    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[0] <= input_size.spatial[0] - 1
            ? (input_size.spatial[0] - off_factor * offset.spatial[0] - 1) / stride.spatial[0] + 1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[1] <= input_size.spatial[1] - 1
            ? (input_size.spatial[1] - off_factor * offset.spatial[1] - 1) / stride.spatial[1] + 1
            : degen_val);
    auto output_range_z = static_cast<cldnn::tensor::value_type>(
        off_factor * offset.spatial[2] <= input_size.spatial[2] - 1
            ? (input_size.spatial[2] - off_factor * offset.spatial[2] - 1) / stride.spatial[2] + 1
            : degen_val);

    return {0, 0, output_range_x, output_range_y, output_range_z};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::exceed_once_data>(const tensor& input_size,
                                                                            const tensor& size,
                                                                            const tensor& offset,
                                                                            const tensor& stride,
                                                                            const tensor& dilation,
                                                                            bool sym_offset,
                                                                            const tensor::value_type& degen_val) {
    auto output_range_exceed_once = calc_sliding_window_output_range<swor_mode::exceed_once>(input_size,
                                                                                             size,
                                                                                             offset,
                                                                                             stride,
                                                                                             dilation,
                                                                                             sym_offset,
                                                                                             degen_val);
    auto output_range_exceed_any_data =
        calc_sliding_window_output_range<swor_mode::any>(input_size, size, offset, stride, dilation, false, degen_val);

    return tensor::min(output_range_exceed_once, output_range_exceed_any_data);
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::max>(const tensor& input_size,
                                                               const tensor& size,
                                                               const tensor& offset,
                                                               const tensor& stride,
                                                               const tensor& dilation,
                                                               bool,
                                                               const tensor::value_type& degen_val) {
    auto output_range_all_sym =
        calc_sliding_window_output_range<swor_mode::all>(input_size, size, offset, stride, dilation, true, degen_val);
    auto output_range_all_asym =
        calc_sliding_window_output_range<swor_mode::all>(input_size, size, offset, stride, dilation, false, degen_val);

    auto output_range_exceed_once_sym = calc_sliding_window_output_range<swor_mode::exceed_once>(input_size,
                                                                                                 size,
                                                                                                 offset,
                                                                                                 stride,
                                                                                                 dilation,
                                                                                                 true,
                                                                                                 degen_val);
    auto output_range_exceed_once_asym = calc_sliding_window_output_range<swor_mode::exceed_once>(input_size,
                                                                                                  size,
                                                                                                  offset,
                                                                                                  stride,
                                                                                                  dilation,
                                                                                                  false,
                                                                                                  degen_val);

    auto output_range_any_sym =
        calc_sliding_window_output_range<swor_mode::any>(input_size, size, offset, stride, dilation, true, degen_val);
    auto output_range_any_asym =
        calc_sliding_window_output_range<swor_mode::any>(input_size, size, offset, stride, dilation, false, degen_val);

    return tensor::max(tensor::max(tensor::max(output_range_all_sym, output_range_all_asym),
                                   tensor::max(output_range_exceed_once_sym, output_range_exceed_once_asym)),
                       tensor::max(output_range_any_sym, output_range_any_asym));
}

/// @brief Calculates minumum needed input range (size) for sliding window to get at least specified @p output_size.
///
/// @param output_size Range/Size of output data (non-padded or treated as valid). Only spatial coordinates are
///                    considered.
/// @param size        Size of sliding window. Only spatial coordinates are considered.
/// @param offset      Offset/Padding of sliding window in input. Only spatial coordinates are considered. Padding/Offset
///                    is applied from both sides of input data: negative value extends/pads data, positive - crops it.
/// @param stride      Horizontal/Vertical stride of sliding in input data.
/// @param dilation    Horizontal/Vertical dilation of sliding window on input data.
/// @param sym_offset  Treat offset as applied on input symmetrically (from both sides). If @c false, the @p offset
///                    is applied only from left/upper side.
/// @param degen_val   If values from calculation are in allowed range, but calculated output size is invalid,
///                    the @p degen_val is returned. Any non-positive value is considered degenerated and will be
///                    switched to value passed in this parameter.
/// @return Input range (size) for sliding window to get equal or greater @p output_size.
inline tensor calc_sliding_window_needed_input_range(const tensor& output_size,
                                                     const tensor& size,
                                                     const tensor& offset,
                                                     const tensor& stride,
                                                     const tensor& dilation = {1, 1, 1, 1},
                                                     bool sym_offset = true,
                                                     const tensor::value_type& degen_val = 0) {
    if (output_size.spatial[0] <= 0 || output_size.spatial[1] <= 0 || output_size.spatial[2] <= 0)
        throw std::invalid_argument("Output data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (stride.spatial[0] <= 0 || stride.spatial[1] <= 0 || stride.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if (dilation.spatial[0] <= 0 || dilation.spatial[1] <= 0 || dilation.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_offset ? 2 : 1;
    tensor wnd_ext_size{0,
                        0,
                        (size.spatial[0] - 1) * dilation.spatial[0] + 1,
                        (size.spatial[1] - 1) * dilation.spatial[1] + 1,
                        (size.spatial[2] - 1) * dilation.spatial[2] + 1};

    auto output_range_x =
        off_factor * offset.spatial[0] + (output_size.spatial[0] - 1) * stride.spatial[0] + wnd_ext_size.spatial[0];
    auto output_range_y =
        off_factor * offset.spatial[1] + (output_size.spatial[1] - 1) * stride.spatial[1] + wnd_ext_size.spatial[1];
    auto output_range_z =
        off_factor * offset.spatial[2] + (output_size.spatial[2] - 1) * stride.spatial[2] + wnd_ext_size.spatial[2];

    if (output_range_x <= 0)
        output_range_x = degen_val;
    if (output_range_y <= 0)
        output_range_y = degen_val;
    if (output_range_z <= 0)
        output_range_z = degen_val;

    return {0, 0, output_range_x, output_range_y, output_range_z};
}

/// @brief Calculates safe needed input upper padding for sliding window to be able to compute at least
/// specified @p output_size.
///
/// @param output_size Range/Size of output data (non-padded or treated as valid). Only spatial coordinates are
///                    considered.
/// @param size        Size of sliding window. Only spatial coordinates are considered.
/// @param offset      Offset/Padding of sliding window in input. Only spatial coordinates are considered. Padding/Offset
///                    is applied from both sides of input data: negative value extends/pads data, positive - crops it.
/// @param stride      Horizontal/Vertical stride of sliding in input data.
/// @param dilation    Horizontal/Vertical dilation of sliding window on input data.
/// @param inverse     Indicate that inverse calculation of needed range should take place (estimation of needed
///                    ouput size when input size is specified). Used in deconvolution (when we switch input calculation
///                    with output calculation).
/// @param degen_val   If values from calculation are in allowed range, but calculated output size is invalid,
///                    the @p degen_val is returned. Any non-positive value is considered degenerated and will be
///                    switched to value passed in this parameter.
/// @return Input upper padding for sliding window to get equal or greater @p output_size. The padding takes into
///         consideration actual value of padding (always extends it) and only works on spatial coordinates of upper
///         padding (rest of padding values are not changed).
inline padding calc_sliding_window_needed_input_padding(const layout& actual_input_layout,
                                                        const tensor& output_size,
                                                        const tensor& size,
                                                        const tensor& offset,
                                                        const tensor& stride,
                                                        const tensor& dilation = {1, 1, 1, 1},
                                                        bool inverse = false,
                                                        const tensor::value_type& degen_val = 0) {
    tensor needed_size;
    if (inverse) {
        needed_size = calc_sliding_window_output_range<swor_mode::max>(output_size,
                                                                       size,
                                                                       offset,
                                                                       stride,
                                                                       dilation,
                                                                       false /* not important */,
                                                                       degen_val);
    } else {
        auto needed_size_sym =
            calc_sliding_window_needed_input_range(output_size, size, offset, stride, dilation, true, degen_val);
        auto needed_size_asym =
            calc_sliding_window_needed_input_range(output_size, size, offset, stride, dilation, false, degen_val);

        needed_size = tensor::max(needed_size_sym, needed_size_asym);
    }

    const auto& actual_data_size = actual_input_layout.size;
    const auto& actual_lpad = actual_input_layout.data_padding.lower_size();
    const auto& actual_upad = actual_input_layout.data_padding.upper_size();

    auto needed_upad = tensor::max(needed_size.sub(actual_data_size), actual_upad);

    return padding(actual_lpad.sizes(),
                   {actual_upad.batch[0],
                    actual_upad.feature[0],
                    needed_upad.spatial[0],
                    needed_upad.spatial[1],
                    needed_upad.spatial[2]});
}

}  // namespace cldnn
