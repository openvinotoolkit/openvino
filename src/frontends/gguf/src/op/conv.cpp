// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

namespace {

// ggml convolutions accumulate and return f32.
ov::Output<ov::Node> as_f32(ov::Output<ov::Node> value) {
    if (value.get_element_type() != ov::element::f32) {
        value = std::make_shared<ov::op::v0::Convert>(value, ov::element::f32);
    }
    return value;
}

std::vector<int64_t> conv_params(const NodeContext& context, size_t count, const char* op) {
    const auto params = context.get_attribute<std::vector<int64_t>>("conv_params");
    FRONT_END_OP_CONVERSION_CHECK(params.size() == count, op, " requires ", count, " conv_params");
    return params;
}

// Static extent of `shape` counted from the innermost axis (1 = innermost).
int64_t inner_dim(const ov::PartialShape& shape, int64_t from_end, const char* op) {
    FRONT_END_OP_CONVERSION_CHECK(shape.rank().is_static() && shape.rank().get_length() >= from_end &&
                                      shape[shape.rank().get_length() - from_end].is_static(),
                                  op,
                                  " requires static operand extents");
    return shape[shape.rank().get_length() - from_end].get_length();
}

}  // namespace

// GGML_OP_CONV_2D_DW: depthwise 2D convolution. Kernel [C, 1, KH, KW], input [N, C, H, W];
// "conv_params" is ggml's {s0, s1, p0, p1, d0, d1} (0 = width, 1 = height).
OutputVector translate_conv_2d_dw(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto p = conv_params(context, 6, "CONV_2D_DW");
    FRONT_END_OP_CONVERSION_CHECK(p[0] > 0 && p[1] > 0 && p[2] >= 0 && p[3] >= 0 && p[4] > 0 && p[5] > 0,
                                  "Invalid CONV_2D_DW parameters");
    // [C, 1, KH, KW] -> GroupConvolution filter [C, 1, 1, KH, KW].
    auto kernel = std::make_shared<ov::op::v0::Unsqueeze>(as_f32(context.get_input(0)),
                                                          ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
    auto res = std::make_shared<ov::op::v1::GroupConvolution>(as_f32(context.get_input(1)),
                                                              kernel,
                                                              ov::Strides{size_t(p[1]), size_t(p[0])},
                                                              ov::CoordinateDiff{std::ptrdiff_t(p[3]), std::ptrdiff_t(p[2])},
                                                              ov::CoordinateDiff{std::ptrdiff_t(p[3]), std::ptrdiff_t(p[2])},
                                                              ov::Strides{size_t(p[5]), size_t(p[4])});
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_OP_CONV_TRANSPOSE_1D: kernel [.., Cin, Cout, K], input [.., N, Cin, L] -> [1, N, Cout, OL];
// "conv_params" is ggml's {s0, p0, d0}.
OutputVector translate_conv_transpose_1d(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto p = conv_params(context, 3, "CONV_TRANSPOSE_1D");
    FRONT_END_OP_CONVERSION_CHECK(p[0] > 0 && p[1] >= 0 && p[2] > 0, "Invalid CONV_TRANSPOSE_1D parameters");
    const auto kernel_shape = context.get_input(0).get_partial_shape();
    const int64_t K = inner_dim(kernel_shape, 1, "CONV_TRANSPOSE_1D");
    const int64_t Cout = inner_dim(kernel_shape, 2, "CONV_TRANSPOSE_1D");
    const int64_t Cin = inner_dim(kernel_shape, 3, "CONV_TRANSPOSE_1D");
    const int64_t L = inner_dim(context.get_input(1).get_partial_shape(), 1, "CONV_TRANSPOSE_1D");
    auto kernel = std::make_shared<ov::op::v1::Reshape>(as_f32(context.get_input(0)),
                                                        ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{Cin, Cout, K}),
                                                        false);
    auto input = std::make_shared<ov::op::v1::Reshape>(as_f32(context.get_input(1)),
                                                       ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{-1, Cin, L}),
                                                       false);
    auto conv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(input,
                                                                      kernel,
                                                                      ov::Strides{size_t(p[0])},
                                                                      ov::CoordinateDiff{std::ptrdiff_t(p[1])},
                                                                      ov::CoordinateDiff{std::ptrdiff_t(p[1])},
                                                                      ov::Strides{size_t(p[2])});
    // [N, Cout, OL] -> ggml's [OL, Cout, N, 1], i.e. OV [1, N, Cout, OL].
    auto res = std::make_shared<ov::op::v0::Unsqueeze>(conv, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_OP_CONV_TRANSPOSE_2D: kernel [Cin, Cout, KH, KW], input [N, Cin, H, W]; "conv_params" is {stride}.
OutputVector translate_conv_transpose_2d(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto p = conv_params(context, 1, "CONV_TRANSPOSE_2D");
    FRONT_END_OP_CONVERSION_CHECK(p[0] > 0, "Invalid CONV_TRANSPOSE_2D stride");
    auto res = std::make_shared<ov::op::v1::ConvolutionBackpropData>(as_f32(context.get_input(1)),
                                                                     as_f32(context.get_input(0)),
                                                                     ov::Strides{size_t(p[0]), size_t(p[0])},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1});
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_OP_CONV_3D: kernel [c * oc, KD, KH, KW], input [c * n, ID, IH, IW] -> [oc * n, OD, OH, OW];
// "conv_params" is ggml's {s0, s1, s2, p0, p1, p2, d0, d1, d2, c, n, oc} (0 = width, 2 = depth).
OutputVector translate_conv_3d(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    const auto p = conv_params(context, 12, "CONV_3D");
    for (size_t i = 0; i < 12; ++i) {
        FRONT_END_OP_CONVERSION_CHECK(i >= 3 && i < 6 ? p[i] >= 0 : p[i] > 0, "Invalid CONV_3D parameters");
    }
    const int64_t c = p[9], n = p[10], oc = p[11];
    const auto kernel_shape = context.get_input(0).get_partial_shape();
    const int64_t KW = inner_dim(kernel_shape, 1, "CONV_3D");
    const int64_t KH = inner_dim(kernel_shape, 2, "CONV_3D");
    const int64_t KD = inner_dim(kernel_shape, 3, "CONV_3D");
    const auto input_shape = context.get_input(1).get_partial_shape();
    const int64_t IW = inner_dim(input_shape, 1, "CONV_3D");
    const int64_t IH = inner_dim(input_shape, 2, "CONV_3D");
    const int64_t ID = inner_dim(input_shape, 3, "CONV_3D");
    auto kernel = std::make_shared<ov::op::v1::Reshape>(
        as_f32(context.get_input(0)),
        ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{oc, c, KD, KH, KW}),
        false);
    auto input = std::make_shared<ov::op::v1::Reshape>(
        as_f32(context.get_input(1)),
        ov::op::v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{n, c, ID, IH, IW}),
        false);
    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          kernel,
                                                          ov::Strides{size_t(p[2]), size_t(p[1]), size_t(p[0])},
                                                          ov::CoordinateDiff{std::ptrdiff_t(p[5]), std::ptrdiff_t(p[4]), std::ptrdiff_t(p[3])},
                                                          ov::CoordinateDiff{std::ptrdiff_t(p[5]), std::ptrdiff_t(p[4]), std::ptrdiff_t(p[3])},
                                                          ov::Strides{size_t(p[8]), size_t(p[7]), size_t(p[6])});
    // [n, oc, OD, OH, OW] -> ggml's [OW, OH, OD, oc * n], i.e. OV [n * oc, OD, OH, OW].
    auto spatial = gather_dims(std::make_shared<ov::op::v3::ShapeOf>(conv, ov::element::i64), {2, 3, 4});
    auto target = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, {1}, {n * oc}), spatial},
        0);
    auto res = std::make_shared<ov::op::v1::Reshape>(conv, target, false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
