// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "onnx_import/core/node.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector prior_box(const Node& node);

OutputVector prior_box_clustered(const Node& node);

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
