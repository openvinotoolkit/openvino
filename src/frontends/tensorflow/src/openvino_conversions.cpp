// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino_conversions.hpp"

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

void convert_nhwc_to_nchw(const std::string& op_name, bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            transpose<0, 3, 1, 2>(node);
        } else if (rank == 5) {
            transpose_3d<0, 4, 1, 2, 3>(node);
        }
    }
}

void convert_nchw_to_nhwc(const std::string& op_name, bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            transpose<0, 2, 3, 1>(node);
        } else if (rank == 5) {
            transpose_3d<0, 2, 3, 4, 1>(node);
        }
    }
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
