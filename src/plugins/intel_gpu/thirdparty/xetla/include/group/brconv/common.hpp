/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "../../common/common.hpp"
#include "../../subgroup/subgroup.hpp"

namespace gpu::xetla::group {
namespace detail {
/// Compute position mapping between input and output activation on each spatial dimension.
template <int32_t val, uint32_t filter_size, uint32_t stride, uint32_t pad,
        uint32_t dilation = 1, bool transpose = false>
constexpr int32_t conv_i2o_spatial_map() {
    // formula for input ==> output dimension
    if constexpr (transpose) {
        return (val - 1) * stride + filter_size - 2 * pad;
    } else {
        return (val + 2 * pad - filter_size) / stride + 1;
    }
}

template <uint32_t filter_size, uint32_t stride, uint32_t pad,
        uint32_t dilation = 1, bool transpose = false>
constexpr int32_t conv_i2o_spatial_map(int32_t val) {
    // formula for input ==> output dimension
    if constexpr (transpose) {
        return (val - 1) * stride + filter_size - 2 * pad;
    } else {
        return (val + 2 * pad - filter_size) / stride + 1;
    }
}

} // namespace detail

template <uint32_t fw_ = 1, uint32_t pad_w_ = 0, uint32_t stride_w_ = 1,
        uint32_t dilation_w_ = 1, uint32_t fh_ = 1, uint32_t pad_h_ = 0,
        uint32_t stride_h_ = 1, uint32_t dilation_h_ = 1>
struct brconv_filter_attr_t {
    static constexpr uint32_t fh = fh_;
    static constexpr uint32_t fw = fw_;
    static constexpr uint32_t pad_h = pad_h_;
    static constexpr uint32_t pad_w = pad_w_;
    static constexpr uint32_t stride_h = stride_h_;
    static constexpr uint32_t stride_w = stride_w_;
    static constexpr uint32_t dilation_h = dilation_h_;
    static constexpr uint32_t dilation_w = dilation_w_;
};

} // namespace gpu::xetla::group
