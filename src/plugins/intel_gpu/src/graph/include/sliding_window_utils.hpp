// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/tensor.hpp"

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"

#include "intel_gpu/runtime/utils.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace ov {
namespace intel_gpu {

using cldnn::tensor;
using cldnn::padding;
using cldnn::layout;

/// @brief Sliding window output range computation mode.
enum class swor_mode {
    // Single modes:
    all,          ///< Range is computed in the way that each sliding window in range is fully contained inside
                  ///< (optionally upper-padded by pad) input data.
    exceed_once,  ///< Range is computed in the way that each except at most one sliding window in range is fully
                  ///< contained inside (optionally upper-padded by pad) input data. The last window may partially
                  ///< exceed (optionally upper-padded by pad) input data range.
    any,          ///< Range is computed in the way that each sliding window in range is fully or at least partially
                  ///< contained inside (optionally upper-padded by pad) input data.
    // Mixed modes:
    exceed_once_data,  ///< Range is computed in the way that each except at most one sliding window in range is fully
                       ///< contained inside (optionally upper-padded by pad) input data. The last window may
                       ///< partially exceed (non-upper-padded) input data range.
                       ///< This mode is effectievely minimum of combination of @c swor_mode::exceed_once mode
                       ///< and @c swor_mode::any mode (with always @c sym_pad = false).
    max                ///< Maximum of all single modes with all cominations of @c sym_pad.
};

/// @brief Calculates output range (size) for sliding window moving on input data range specified by @p input_size.
///
/// @param input_size Range/Size of input data (non-padded or treated as valid). Only spatial coordinates are
///                   considered.
/// @param size       Size of sliding window. Only spatial coordinates are considered.
/// @param pad     pad/Padding of sliding window in input. Only spatial coordinates are considered. Padding/pad
///                   is applied from both sides of input data: negative value extends/pads data, positive - crops it.
/// @param stride     Horizontal/Vertical stride of sliding in input data.
/// @param dilation   Horizontal/Vertical dilation of sliding window on input data.
/// @param sym_pad Treat pad as applied on input symmetrically (from both sides). If @c false, the @p pad
///                   is applied only from left/upper side.
/// @param degen_val  If values from calculation are in allowed range, but calculated output size is invalid,
///                   the @p degen_val is returned. Any non-positive value is considered degenerated and will be
///                   switched to value passed in this parameter.
/// @return Output range (size) of sliding window. Only spatial dimensions are valid (rest is 0).
template <swor_mode RangeMode = swor_mode::all>
tensor calc_sliding_window_output_range(const tensor& input_size,
                                        const tensor& size,
                                        const ov::CoordinateDiff& pad,
                                        const ov::Strides& stride,
                                        const ov::Strides& dilation = {1, 1, 1, 1},
                                        bool sym_pad = true,
                                        const tensor::value_type& degen_val = 0);

/// @brief Fall-back implementation.
template <swor_mode RangeMode>
tensor calc_sliding_window_output_range(const tensor&,
                                        const tensor&,
                                        const tensor&,
                                        const ov::Strides&,
                                        const tensor&,
                                        bool,
                                        const tensor::value_type&) {
    static_assert(cldnn::meta::always_false<cldnn::meta::val_tuple<swor_mode, RangeMode>>::value,
                  "Sliding window output range calculation mode is not supported. Please implement specialization "
                  "for new swor_mode.");

    return tensor();
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::all>(const tensor& input_size,
                                                               const tensor& size,
                                                               const ov::CoordinateDiff& pad,
                                                               const ov::Strides& stride,
                                                               const ov::Strides& dilation,
                                                               bool sym_pad,
                                                               const tensor::value_type& degen_val) {
    if (input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0 || input_size.spatial[2] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (std::any_of(stride.begin(), stride.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window strides must be positive (>= 1).");
    if (std::any_of(dilation.begin(), dilation.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_pad ? -2 : -1;
    auto stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    auto stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    auto stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    tensor::value_type dilation_z = dilation.size() >= 3 ? static_cast<int32_t>(dilation[dilation.size() - 3]) : 1;
    tensor::value_type dilation_y = dilation.size() >= 2 ? static_cast<int32_t>(dilation[dilation.size() - 2]) : 1;
    tensor::value_type dilation_x = dilation.size() >= 1 ? static_cast<int32_t>(dilation[dilation.size() - 1]) : 1;

    auto pad_z = pad.size() >= 3 ? pad[pad.size() - 3] : 0;
    auto pad_y = pad.size() >= 2 ? pad[pad.size() - 2] : 0;
    auto pad_x = pad.size() >= 1 ? pad[pad.size() - 1] : 0;

    tensor wnd_ext_size{0,
                        0,
                        (size.spatial[0] - 1) * dilation_x + 1,
                        (size.spatial[1] - 1) * dilation_y + 1,
                        (size.spatial[2] - 1) * dilation_z + 1};

    // wes = (size - 1) * dilation + 1
    // lpos(i) = -pad + i * stride + wes - 1,   for i = 0, 1, ...
    //
    // output_range = max {i | lpos(i) < input_size + pad} + 1,      if sym_pad is true
    // output_range = max {i | lpos(i) < input_size} + 1,            if sym_pad is false
    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_x + wnd_ext_size.spatial[0] <= input_size.spatial[0]
            ? (input_size.spatial[0] - off_factor * pad_x - wnd_ext_size.spatial[0]) / stride_x + 1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_y + wnd_ext_size.spatial[1] <= input_size.spatial[1]
            ? (input_size.spatial[1] - off_factor * pad_y - wnd_ext_size.spatial[1]) / stride_y + 1
            : degen_val);
    auto output_range_z = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_z + wnd_ext_size.spatial[2] <= input_size.spatial[2]
            ? (input_size.spatial[2] - off_factor * pad_z - wnd_ext_size.spatial[2]) / stride_z + 1
            : degen_val);

     return {0, 0, output_range_x, output_range_y, output_range_z};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::exceed_once>(const tensor& input_size,
                                                                       const tensor& size,
                                                                       const ov::CoordinateDiff& pad,
                                                                       const ov::Strides& stride,
                                                                       const ov::Strides& dilation,
                                                                       bool sym_pad,
                                                                       const tensor::value_type& degen_val) {
    if (input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0 || input_size.spatial[2] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (std::any_of(stride.begin(), stride.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window strides must be positive (>= 1).");
    if (std::any_of(dilation.begin(), dilation.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    int64_t off_factor = sym_pad ? -2 : -1;
    int64_t stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    int64_t stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    int64_t stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    tensor::value_type dilation_z = dilation.size() >= 3 ? static_cast<int32_t>(dilation[dilation.size() - 3]) : 1;
    tensor::value_type dilation_y = dilation.size() >= 2 ? static_cast<int32_t>(dilation[dilation.size() - 2]) : 1;
    tensor::value_type dilation_x = dilation.size() >= 1 ? static_cast<int32_t>(dilation[dilation.size() - 1]) : 1;

    int64_t pad_z = pad.size() >= 3 ? pad[pad.size() - 3] : 0;
    int64_t pad_y = pad.size() >= 2 ? pad[pad.size() - 2] : 0;
    int64_t pad_x = pad.size() >= 1 ? pad[pad.size() - 1] : 0;

    tensor extend{0,
                  0,
                  std::max<tensor::value_type>((size.spatial[0] - 1) * dilation_x + 1, stride_x),
                  std::max<tensor::value_type>((size.spatial[1] - 1) * dilation_y + 1, stride_y),
                  std::max<tensor::value_type>((size.spatial[2] - 1) * dilation_z + 1, stride_z)};


    // wes = (size - 1) * dilation + 1
    // fpos(i) = -pad + i * stride,             for i = 0, 1, ...
    // lpos(i) = -pad + i * stride + wes - 1,   for i = 0, 1, ...
    //
    // output_range = max {i | lpos(i) < input_size + pad - 1 and fpos(i + 1) < input_size + pad} + 2,   if
    // sym_pad is true output_range = max {i | lpos(i) < input_size - 1          and fpos(i + 1) < input_size} + 2,
    // if sym_pad is false
    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_x + extend.spatial[0] <= input_size.spatial[0] + stride_x - 1
            ? (input_size.spatial[0] - off_factor * pad_x - extend.spatial[0] + stride_x - 1) /
                      stride_x +
                  1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_y + extend.spatial[1] <= input_size.spatial[1] + stride_y - 1
            ? (input_size.spatial[1] - off_factor * pad_y - extend.spatial[1] + stride_y - 1) /
                      stride_y +
                  1
            : degen_val);
    auto output_range_z = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_z + extend.spatial[2] <= input_size.spatial[2] + stride_z - 1
            ? (input_size.spatial[2] - off_factor * pad_z - extend.spatial[2] + stride_z - 1) /
                      stride_z +
                  1
            : degen_val);

    return {0, 0, output_range_x, output_range_y, output_range_z};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::any>(const tensor& input_size,
                                                               const tensor& size,
                                                               const ov::CoordinateDiff& pad,
                                                               const ov::Strides& stride,
                                                               const ov::Strides& dilation,
                                                               bool sym_pad,
                                                               const tensor::value_type& degen_val) {
    if (input_size.spatial[0] <= 0 || input_size.spatial[1] <= 0 || input_size.spatial[2] <= 0)
        throw std::invalid_argument("Input data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (std::any_of(stride.begin(), stride.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if (std::any_of(dilation.begin(), dilation.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    auto stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    auto stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    auto pad_z = pad.size() >= 3 ? pad[pad.size() - 3] : 0;
    auto pad_y = pad.size() >= 2 ? pad[pad.size() - 2] : 0;
    auto pad_x = pad.size() >= 1 ? pad[pad.size() - 1] : 0;

    auto off_factor = sym_pad ? -2 : -1;

    // fpos(i) = -pad + i * stride,             for i = 0, 1, ...
    //
    // output_range = max {i | fpos(i) < input_size + pad} + 1,      if sym_pad is true
    // output_range = max {i | fpos(i) < input_size} + 1,            if sym_pad is false
    auto output_range_x = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_x <= input_size.spatial[0] - 1
            ? (input_size.spatial[0] - off_factor * pad_x - 1) / stride_x + 1
            : degen_val);
    auto output_range_y = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_y <= input_size.spatial[1] - 1
            ? (input_size.spatial[1] - off_factor * pad_y - 1) / stride_y + 1
            : degen_val);
    auto output_range_z = static_cast<cldnn::tensor::value_type>(
        off_factor * pad_z <= input_size.spatial[2] - 1
            ? (input_size.spatial[2] - off_factor * pad_z - 1) / stride_z + 1
            : degen_val);

    return {0, 0, output_range_x, output_range_y, output_range_z};
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::exceed_once_data>(const tensor& input_size,
                                                                            const tensor& size,
                                                                            const ov::CoordinateDiff& pad,
                                                                            const ov::Strides& stride,
                                                                            const ov::Strides& dilation,
                                                                            bool sym_pad,
                                                                            const tensor::value_type& degen_val) {
    auto output_range_exceed_once = calc_sliding_window_output_range<swor_mode::exceed_once>(input_size,
                                                                                             size,
                                                                                             pad,
                                                                                             stride,
                                                                                             dilation,
                                                                                             sym_pad,
                                                                                             degen_val);
    auto output_range_exceed_any_data =
        calc_sliding_window_output_range<swor_mode::any>(input_size, size, pad, stride, dilation, false, degen_val);

    return tensor::min(output_range_exceed_once, output_range_exceed_any_data);
}

template <>
inline tensor calc_sliding_window_output_range<swor_mode::max>(const tensor& input_size,
                                                               const tensor& size,
                                                               const ov::CoordinateDiff& pad,
                                                               const ov::Strides& stride,
                                                               const ov::Strides& dilation,
                                                               bool,
                                                               const tensor::value_type& degen_val) {
    auto output_range_all_sym =
        calc_sliding_window_output_range<swor_mode::all>(input_size, size, pad, stride, dilation, true, degen_val);
    auto output_range_all_asym =
        calc_sliding_window_output_range<swor_mode::all>(input_size, size, pad, stride, dilation, false, degen_val);

    auto output_range_exceed_once_sym = calc_sliding_window_output_range<swor_mode::exceed_once>(input_size,
                                                                                                 size,
                                                                                                 pad,
                                                                                                 stride,
                                                                                                 dilation,
                                                                                                 true,
                                                                                                 degen_val);
    auto output_range_exceed_once_asym = calc_sliding_window_output_range<swor_mode::exceed_once>(input_size,
                                                                                                  size,
                                                                                                  pad,
                                                                                                  stride,
                                                                                                  dilation,
                                                                                                  false,
                                                                                                  degen_val);

    auto output_range_any_sym =
        calc_sliding_window_output_range<swor_mode::any>(input_size, size, pad, stride, dilation, true, degen_val);
    auto output_range_any_asym =
        calc_sliding_window_output_range<swor_mode::any>(input_size, size, pad, stride, dilation, false, degen_val);

    return tensor::max(tensor::max(tensor::max(output_range_all_sym, output_range_all_asym),
                                   tensor::max(output_range_exceed_once_sym, output_range_exceed_once_asym)),
                       tensor::max(output_range_any_sym, output_range_any_asym));
}

/// @brief Calculates minumum needed input range (size) for sliding window to get at least specified @p output_size.
///
/// @param output_size Range/Size of output data (non-padded or treated as valid). Only spatial coordinates are
///                    considered.
/// @param size        Size of sliding window. Only spatial coordinates are considered.
/// @param pad      pad/Padding of sliding window in input. Only spatial coordinates are considered. Padding/pad
///                    is applied from both sides of input data: negative value extends/pads data, positive - crops it.
/// @param stride      Horizontal/Vertical stride of sliding in input data.
/// @param dilation    Horizontal/Vertical dilation of sliding window on input data.
/// @param sym_pad  Treat pad as applied on input symmetrically (from both sides). If @c false, the @p pad
///                    is applied only from left/upper side.
/// @param degen_val   If values from calculation are in allowed range, but calculated output size is invalid,
///                    the @p degen_val is returned. Any non-positive value is considered degenerated and will be
///                    switched to value passed in this parameter.
/// @return Input range (size) for sliding window to get equal or greater @p output_size.
inline tensor calc_sliding_window_needed_input_range(const tensor& output_size,
                                                     const tensor& size,
                                                     const ov::CoordinateDiff& pad,
                                                     const ov::Strides& stride,
                                                     const ov::Strides& dilation = {1, 1, 1, 1},
                                                     bool sym_pad = true,
                                                     const tensor::value_type& degen_val = 0) {
    if (output_size.spatial[0] <= 0 || output_size.spatial[1] <= 0 || output_size.spatial[2] <= 0)
        throw std::invalid_argument("Output data spatial sizes must be positive (>= 1).");
    if (size.spatial[0] <= 0 || size.spatial[1] <= 0 || size.spatial[2] <= 0)
        throw std::invalid_argument("Sliding window spatial sizes must be positive (>= 1).");
    if (std::any_of(stride.begin(), stride.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window h/v strides must be positive (>= 1).");
    if (std::any_of(dilation.begin(), dilation.end(), [](size_t v) { return v <= 0; }))
        throw std::invalid_argument("Sliding window h/v input dialations must be positive (>= 1).");

    auto off_factor = sym_pad ? -2 : -1;
    auto stride_z = stride.size() >= 3 ? stride[stride.size() - 3] : 1;
    auto stride_y = stride.size() >= 2 ? stride[stride.size() - 2] : 1;
    auto stride_x = stride.size() >= 1 ? stride[stride.size() - 1] : 1;

    tensor::value_type dilation_z = dilation.size() >= 3 ? static_cast<int32_t>(dilation[dilation.size() - 3]) : 1;
    tensor::value_type dilation_y = dilation.size() >= 2 ? static_cast<int32_t>(dilation[dilation.size() - 2]) : 1;
    tensor::value_type dilation_x = dilation.size() >= 1 ? static_cast<int32_t>(dilation[dilation.size() - 1]) : 1;

    auto pad_z = pad.size() >= 3 ? pad[pad.size() - 3] : 0;
    auto pad_y = pad.size() >= 2 ? pad[pad.size() - 2] : 0;
    auto pad_x = pad.size() >= 1 ? pad[pad.size() - 1] : 0;

    tensor wnd_ext_size{0,
                        0,
                        (size.spatial[0] - 1) * dilation_x + 1,
                        (size.spatial[1] - 1) * dilation_y + 1,
                        (size.spatial[2] - 1) * dilation_z + 1};

    auto output_range_x =
        off_factor * pad_x + (output_size.spatial[0] - 1) * stride_x + wnd_ext_size.spatial[0];
    auto output_range_y =
        off_factor * pad_y + (output_size.spatial[1] - 1) * stride_y + wnd_ext_size.spatial[1];
    auto output_range_z =
        off_factor * pad_z + (output_size.spatial[2] - 1) * stride_z + wnd_ext_size.spatial[2];

    if (output_range_x <= 0)
        output_range_x = degen_val;
    if (output_range_y <= 0)
        output_range_y = degen_val;
    if (output_range_z <= 0)
        output_range_z = degen_val;

    return {0,
            0,
            static_cast<tensor::value_type>(output_range_x),
            static_cast<tensor::value_type>(output_range_y),
            static_cast<tensor::value_type>(output_range_z)};
}

/// @brief Calculates safe needed input upper padding for sliding window to be able to compute at least
/// specified @p output_size.
///
/// @param output_size Range/Size of output data (non-padded or treated as valid). Only spatial coordinates are
///                    considered.
/// @param size        Size of sliding window. Only spatial coordinates are considered.
/// @param pad         Padding of sliding window in input. Only spatial coordinates are considered. Padding/pad
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
                                                        const ov::CoordinateDiff& pad,
                                                        const ov::Strides& stride,
                                                        const ov::Strides& dilation = {1, 1},
                                                        bool inverse = false,
                                                        const tensor::value_type& degen_val = 0) {
    tensor needed_size;
    if (inverse) {
        needed_size = calc_sliding_window_output_range<swor_mode::max>(output_size,
                                                                       size,
                                                                       pad,
                                                                       stride,
                                                                       dilation,
                                                                       false /* not important */,
                                                                       degen_val);
    } else {
        auto needed_size_sym =
            calc_sliding_window_needed_input_range(output_size, size, pad, stride, dilation, true, degen_val);
        auto needed_size_asym =
            calc_sliding_window_needed_input_range(output_size, size, pad, stride, dilation, false, degen_val);

        needed_size = tensor::max(needed_size_sym, needed_size_asym);
    }

    const auto& actual_data_size = actual_input_layout.get_tensor();
    const auto& actual_lpad = actual_input_layout.data_padding._lower_size;
    const auto& actual_upad = actual_input_layout.data_padding._upper_size;

    auto needed_upad = needed_size.sub(actual_data_size);

    auto spatial_rank = actual_input_layout.get_spatial_rank();
    OPENVINO_ASSERT(spatial_rank > 0 && spatial_rank <= 3);
    if (spatial_rank >= 3)
        return padding({actual_lpad[0], actual_lpad[1], actual_lpad[2], actual_lpad[3], actual_lpad[4]},
                       {actual_upad[0],
                        actual_upad[1],
                        std::max(needed_upad.spatial[2], actual_upad[2]),
                        std::max(needed_upad.spatial[1], actual_upad[3]),
                        std::max(needed_upad.spatial[0], actual_upad[4])});
    else if (spatial_rank >= 2)
        return padding({actual_lpad[0], actual_lpad[1], actual_lpad[2], actual_lpad[3]},
                       {actual_upad[0],
                        actual_upad[1],
                        std::max(needed_upad.spatial[1], actual_upad[2]),
                        std::max(needed_upad.spatial[0], actual_upad[3])});
    else
        return padding({actual_lpad[0], actual_lpad[1], actual_lpad[2]},
                       {actual_upad[0],
                        actual_upad[1],
                        std::max(needed_upad.spatial[0], actual_upad[2])});
}

}  // namespace intel_gpu
}  // namespace ov
