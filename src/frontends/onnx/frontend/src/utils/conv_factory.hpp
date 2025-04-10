// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace conv_factory {
std::shared_ptr<ov::op::Op> make_ng_convolution(const ov::Output<ov::Node>& data,
                                                const ov::Output<ov::Node>& filters,
                                                const ov::Strides& strides,
                                                const ov::Strides& dilations,
                                                const ov::CoordinateDiff& padding_below,
                                                const ov::CoordinateDiff& padding_above,
                                                int64_t groups,
                                                const ov::op::PadType& auto_pad);
}  // namespace conv_factory
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
