// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_conversions.hpp"

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tf {

void NHWCtoNCHW(const std::string& op_name, bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            Transpose<0, 3, 1, 2>(node);
        } else if (rank == 5) {
            Transpose3D<0, 4, 1, 2, 3>(node);
        }
        SetTracingInfo(op_name, node);
    }
}

void NCHWtoNHWC(const std::string& op_name, bool need_convert, ov::Output<ov::Node>& node) {
    if (need_convert) {
        auto rank = node.get_shape().size();
        if (rank == 4) {
            Transpose<0, 2, 3, 1>(node);
        } else if (rank == 5) {
            Transpose3D<0, 2, 3, 4, 1>(node);
        }
        SetTracingInfo(op_name, node);
    }
}

}  // namespace tf
}  // namespace frontend
}  // namespace ov
