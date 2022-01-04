// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "onnx_import/core/node.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/experimental_detectron_roi_feature.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector experimental_detectron_roi_feature_extractor(const Node& node);
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
