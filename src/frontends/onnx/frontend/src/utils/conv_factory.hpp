// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace conv_factory {
std::shared_ptr<ov::op::Op> make_ng_convolution(const Output<ngraph::Node>& data,
                                                const Output<ngraph::Node>& filters,
                                                const ngraph::Strides& strides,
                                                const ngraph::Strides& dilations,
                                                const ngraph::CoordinateDiff& padding_below,
                                                const ngraph::CoordinateDiff& padding_above,
                                                int64_t groups,
                                                const ngraph::op::PadType& auto_pad);
}  // namespace conv_factory
}  // namespace onnx_import
}  // namespace ngraph
