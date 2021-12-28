// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_conversions.hpp"

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

void convert_nhwc_to_nchw(bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            Transpose<0, 3, 1, 2>(node);
        } else if (rank == 5) {
            Transpose3D<0, 4, 1, 2, 3>(node);
        }
    }
}

void convert_nchw_to_nhwc(bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            Transpose<0, 2, 3, 1>(node);
        } else if (rank == 5) {
            Transpose3D<0, 2, 3, 4, 1>(node);
        }
    }
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
