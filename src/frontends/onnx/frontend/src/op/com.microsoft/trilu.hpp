// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/trilu.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector trilu(const Node& node) {
    return set_14::trilu(node);
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
