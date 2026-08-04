// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_SSM_CONV: depthwise 1D causal convolution over the conv-state window (SSM / Mamba-style
// models, e.g. qwen3next). Implemented as a GroupConvolution with groups == channels.
OutputVector translate_ssm_conv(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto sx = context.get_input(0);  // conv state + input: OV [1, n_s, d_inner, ncs]
    auto c = context.get_input(1);   // conv1d weight:      OV [1, 1, d_inner, d_conv]

    auto sx_shape = context.get_input_shape(0).to_shape();
    auto c_shape = context.get_input_shape(1).to_shape();

    int64_t n_s = sx_shape[1];
    int64_t d_inner = sx_shape[2];
    int64_t d_conv = c_shape[3];

    // The conv-window length ncs (= n_t + d_conv - 1) and the token count n_t are token-dependent: the
    // stateful model is compiled once and reused across token counts, so keep those axes dynamic (-1)
    // rather than baking the convert-time value. n_s/d_inner/d_conv are structurally static.
    // [1, n_s, d_inner, ncs] -> [n_s, d_inner, ncs] (ncs dynamic) for 1D GroupConvolution.
    auto sx_new_shape = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{n_s, d_inner, -1});
    auto sx_reshaped = std::make_shared<ov::op::v1::Reshape>(sx, sx_new_shape, false);

    // Filter [groups, out/groups, in/groups, kernel] = [d_inner, 1, 1, d_conv].
    auto c_new_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{d_inner, 1, 1, d_conv});
    auto c_reshaped = std::make_shared<ov::op::v1::Reshape>(c, c_new_shape, false);

    auto conv = std::make_shared<ov::op::v1::GroupConvolution>(
        sx_reshaped, c_reshaped, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});

    // [n_s, d_inner, n_t] -> [n_s, n_t, d_inner]
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{0, 2, 1});
    auto transposed = std::make_shared<ov::op::v1::Transpose>(conv, perm);

    // [1, n_s, n_t, d_inner] with the token axis n_t dynamic (-1).
    auto out_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, n_s, -1, d_inner});
    auto res = std::make_shared<ov::op::v1::Reshape>(transposed, out_shape, false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
