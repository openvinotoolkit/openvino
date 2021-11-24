// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tensorflow_frontend/utility.hpp>
#include <tensorflow_frontend/extension.hpp>
#include <tensorflow_frontend/frontend.hpp>

#include "utils.hpp"
#include "ngraph_conversions.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace tf {
namespace op {

#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)
OP_CONVERTER(translate_fused_conv_2d_op);
OP_CONVERTER(translate_fused_mat_mul_op);
OP_CONVERTER(translate_fused_batch_norm_op);
OP_CONVERTER(translate_depthwise_conv_2d_native_op);
OP_CONVERTER(translate_retval_op);

}
}
}
}


