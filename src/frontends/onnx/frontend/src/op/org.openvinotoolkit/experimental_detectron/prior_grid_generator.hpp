// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "ngraph/node.hpp"
#include "ngraph/op/experimental_detectron_prior_grid_generator.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector experimental_detectron_prior_grid_generator(const Node& node);

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
