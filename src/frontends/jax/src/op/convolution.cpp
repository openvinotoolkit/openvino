// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

bool need_transpose(const std::vector<int64_t>& spec) {
    int64_t num = spec[0];
    for (size_t i = 1; i < spec.size(); i++) {
        if (num > spec[i]) {
            return true;
        }
        num = spec[i];
    }
    return false;
}

OutputVector translate_convolution(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    Output<Node> input = context.get_input(0);
    Output<Node> kernel = context.get_input(1);

    auto dimension_numbers = context.const_named_param<std::vector<std::vector<int64_t>>>("dimension_numbers");
    auto strides = context.const_named_param<Strides>("window_strides");
    auto padding = context.const_named_param<std::vector<std::vector<int64_t>>>("padding");
    auto dilations = context.const_named_param<Strides>("lhs_dilation");
    auto kernel_dilation = context.const_named_param<std::vector<int64_t>>("rhs_dilation");
    auto feature_group_count = context.const_named_param<int64_t>("feature_group_count");
    auto batch_group_count = context.const_named_param<int64_t>("batch_group_count");
    size_t spatial_dim = strides.size();

    JAX_OP_CONVERSION_CHECK(dimension_numbers.size() == 3,
                            "Internal error: dimension_numbers must have 3 vectors but actually got ",
                            dimension_numbers.size());
    auto lhs_spec = dimension_numbers[0];
    auto rhs_spec = dimension_numbers[1];
    auto out_spec = dimension_numbers[2];
    JAX_OP_CONVERSION_CHECK(lhs_spec.size() == rhs_spec.size() && lhs_spec.size() == out_spec.size(),
                            "Internal error: specs in dimension_numbers must have the same size, but "
                            "got lhs_spec.size() = ",
                            lhs_spec.size(),
                            ", rhs_spec.size() = ",
                            rhs_spec.size(),
                            ", out_spec.size() = ",
                            out_spec.size());
    JAX_OP_CONVERSION_CHECK(lhs_spec.size() == 4, "Internal error: specs in dimension_numbers must have 4 elements.");

    JAX_OP_CONVERSION_CHECK(spatial_dim == 2 || spatial_dim == 3,
                            "Internal error: only 2D and 3D convolutions are supported.");
    JAX_OP_CONVERSION_CHECK(padding.size() == 2, "Inconsistent model: padding must have 2 vectors.");
    JAX_OP_CONVERSION_CHECK(padding[0].size() == spatial_dim,
                            "Inconsistent model: padding vector must contain elements equal to "
                            "doubled spatial dimensions ");
    JAX_OP_CONVERSION_CHECK(dilations.size() == spatial_dim,
                            "Inconsistent model: input_dilation vector must contain elements equal to "
                            "spatial dimensions ");
    JAX_OP_CONVERSION_CHECK(kernel_dilation.size() == spatial_dim,
                            "Inconsistent model: kernel_dilation vector must contain elements equal to "
                            "spatial dimensions ");
    JAX_OP_CONVERSION_CHECK(batch_group_count == 1,
                            "Convolution is supported only with batch_group_count equal to one yet.");

    // Tranpose input and kernel to NCHW format
    if (need_transpose(lhs_spec)) {
        auto input_transpose_order = std::make_shared<v0::Constant>(element::i64, Shape{lhs_spec.size()}, lhs_spec);
        input = std::make_shared<v1::Transpose>(input, input_transpose_order);
    }
    if (need_transpose(rhs_spec)) {
        auto kernel_transpose_order = std::make_shared<v0::Constant>(element::i64, Shape{rhs_spec.size()}, rhs_spec);
        kernel = std::make_shared<v1::Transpose>(kernel, kernel_transpose_order);
    }

    CoordinateDiff pads_begin(spatial_dim);
    CoordinateDiff pads_end(spatial_dim);
    for (size_t ind = 0; ind < spatial_dim; ++ind) {
        pads_begin[ind] = padding[ind][0];
        pads_end[ind] = padding[ind][1];
    }

    Output<Node> res;
    if (feature_group_count == 1) {
        res = std::make_shared<v1::Convolution>(input, kernel, strides, pads_begin, pads_end, dilations);
    } else {
        // use group convolution
        // for this, reformat kernel to have [GROUPS, C_OUT, C_IN, Z, Y, X]
        // 1. compute a part of kernel shape [C_IN, Z, Y, X]
        auto kernel_shape = std::make_shared<v3::ShapeOf>(kernel, element::i64);
        auto start = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
        auto step = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
        auto stop = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, std::numeric_limits<int>::max());
        auto kernel_shape_part = std::make_shared<v8::Slice>(kernel_shape, start, stop, step);
        // 2. create a new shape of the kernel [GROUPS, -1, C_IN, Z, Y, X]
        auto feature_group_const = std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, feature_group_count);
        auto minus_one = std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, -1);
        auto new_shape =
            std::make_shared<v0::Concat>(OutputVector{feature_group_const, minus_one, kernel_shape_part}, 0);
        kernel = std::make_shared<v1::Reshape>(kernel, new_shape, false);
        // 3. compute group convolution using reformatted kernel.
        res = std::make_shared<v1::GroupConvolution>(input, kernel, strides, pads_begin, pads_end, dilations);
    }

    if (need_transpose(out_spec)) {
        std::vector<int64_t> out_transpose_vector(out_spec.size());
        for (size_t i = 0; i < out_spec.size(); i++) {
            out_transpose_vector[out_spec[i]] = i;
        }
        auto output_transpose_order =
            std::make_shared<v0::Constant>(element::i64, Shape{out_transpose_vector.size()}, out_transpose_vector);
        res = std::make_shared<v1::Transpose>(res, output_transpose_order);
    }

    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov