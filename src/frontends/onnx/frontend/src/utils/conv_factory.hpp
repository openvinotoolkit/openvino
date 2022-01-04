// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "onnx_import/core/node.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace onnx_import {
namespace conv_factory {
std::shared_ptr<ov::op::Op> make_ng_convolution(const Output<ov::Node>& data,
                                                const Output<ov::Node>& filters,
                                                const Strides& strides,
                                                const Strides& dilations,
                                                const CoordinateDiff& padding_below,
                                                const CoordinateDiff& padding_above,
                                                int64_t groups,
                                                const ov::op::PadType& auto_pad);
}  // namespace conv_factory
}  // namespace onnx_import
}  // namespace ov
