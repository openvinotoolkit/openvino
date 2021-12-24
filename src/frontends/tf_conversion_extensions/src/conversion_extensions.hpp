// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/tensorflow/extension/conversion.hpp>
#include <openvino/frontend/tensorflow/frontend.hpp>

#include "ngraph_conversions.hpp"
#include "openvino/core/node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)
OP_CONVERTER(translate_fused_conv_2d_op);
OP_CONVERTER(translate_fused_mat_mul_op);
OP_CONVERTER(translate_fused_batch_norm_op);
OP_CONVERTER(translate_depthwise_conv_2d_native_op);
OP_CONVERTER(translate_retval_op);

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
