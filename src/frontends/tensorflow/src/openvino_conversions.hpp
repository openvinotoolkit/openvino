// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

std::shared_ptr<ov::opset8::Transpose> make_transpose(const ov::Output<ov::Node>& arg,
                                                      const ov::AxisVector& input_order);

namespace detail {
template <typename T>
void convert_nhwc_to_hw(const std::vector<T>& src, std::vector<size_t>& dst) {
    if (dst.size() >= 2) {
        dst[0] = src[1];
        dst[1] = src[2];
    }
    if (dst.size() >= 3) {
        dst[2] = src[3];
    }
}

template <typename T>
void convert_nchw_to_hw(const std::vector<T>& src, std::vector<size_t>& dst) {
    if (dst.size() >= 2) {
        dst[0] = src[2];
        dst[1] = src[3];
    }
    if (dst.size() >= 3) {
        dst[2] = src[4];
    }
}
}  // namespace detail

void convert_nhwc_to_nchw(bool need_convert, ov::Output<ov::Node>& node, ov::Rank input_rank = ov::Rank::dynamic());

void convert_nchw_to_nhwc(bool need_convert, ov::Output<ov::Node>& node, ov::Rank input_rank = ov::Rank::dynamic());

template <typename T>
void convert_nhwc_to_hw(bool is_nhwc, const std::vector<T>& src, std::vector<size_t>& dst) {
    if (is_nhwc) {
        detail::convert_nhwc_to_hw(src, dst);
    } else {
        detail::convert_nchw_to_hw(src, dst);
    }
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
