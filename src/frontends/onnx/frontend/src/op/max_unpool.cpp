// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_9 {
ov::OutputVector max_unpool(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2, 3);
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs[0];
    const auto& indices = inputs[1];

    const auto kernel_shape = node.get_attribute_value<std::vector<int64_t>>("kernel_shape");
    const auto spatial_rank = kernel_shape.size();
    const auto strides =
        node.get_attribute_value<std::vector<int64_t>>("strides", std::vector<int64_t>(spatial_rank, 1));
    const auto pads = node.get_attribute_value<std::vector<int64_t>>("pads", std::vector<int64_t>(spatial_rank * 2, 0));

    const auto is_positive = [](int64_t v) {
        return v > 0;
    };
    CHECK_VALID_NODE(node,
                     spatial_rank > 0 && std::all_of(kernel_shape.begin(), kernel_shape.end(), is_positive),
                     "MaxUnpool 'kernel_shape' attribute must be non-empty and positive.");
    CHECK_VALID_NODE(node,
                     strides.size() == spatial_rank && std::all_of(strides.begin(), strides.end(), is_positive),
                     "MaxUnpool 'strides' attribute must have ",
                     spatial_rank,
                     " positive elements. Got: ",
                     strides.size());
    CHECK_VALID_NODE(node,
                     pads.size() == spatial_rank * 2,
                     "MaxUnpool 'pads' attribute must have ",
                     spatial_rank * 2,
                     " elements. Got: ",
                     pads.size());
    const auto data_rank = data.get_partial_shape().rank();
    CHECK_VALID_NODE(node,
                     data_rank.is_dynamic() || data_rank.get_length() == static_cast<int64_t>(spatial_rank + 2),
                     "MaxUnpool input rank must be 'kernel_shape' size + 2. Got rank: ",
                     data_rank,
                     ", 'kernel_shape' size: ",
                     spatial_rank);

    ov::Output<ov::Node> output_shape;
    if (common::is_input_valid(node, 2)) {
        output_shape = inputs[2];
    } else {
        // out[i] = (in[i] - 1) * strides[i] + kernel_shape[i] - pads_begin[i] - pads_end[i]
        std::vector<int64_t> shift(spatial_rank);
        for (size_t i = 0; i < spatial_rank; ++i) {
            shift[i] = kernel_shape[i] - strides[i] - pads[i] - pads[i + spatial_rank];
        }
        const auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
        const auto step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto batch_channels = std::make_shared<v8::Slice>(data_shape,
                                                                v0::Constant::create(ov::element::i64, {1}, {0}),
                                                                v0::Constant::create(ov::element::i64, {1}, {2}),
                                                                step);
        const auto spatial_dims =
            std::make_shared<v8::Slice>(data_shape,
                                        v0::Constant::create(ov::element::i64, {1}, {2}),
                                        v0::Constant::create(ov::element::i64, {1}, {std::numeric_limits<int64_t>::max()}),
                                        step);
        const auto scaled =
            std::make_shared<v1::Multiply>(spatial_dims,
                                           v0::Constant::create(ov::element::i64, {spatial_rank}, strides));
        const auto out_spatial =
            std::make_shared<v1::Add>(scaled, v0::Constant::create(ov::element::i64, {spatial_rank}, shift));
        output_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch_channels, out_spatial}, 0);
    }

    // Indices address the whole flattened output (N x C x D1 x ... x Dn)
    const auto zero = std::make_shared<v1::ConvertLike>(v0::Constant::create(ov::element::f32, {}, {0}), data);
    const auto total_size =
        std::make_shared<v1::ReduceProd>(output_shape, v0::Constant::create(ov::element::i64, {1}, {0}), true);
    const auto zeros = std::make_shared<v3::Broadcast>(zero, total_size);
    const auto flat_shape = v0::Constant::create(ov::element::i64, {1}, {-1});
    const auto flat_data = std::make_shared<v1::Reshape>(data, flat_shape, false);
    const auto flat_indices = std::make_shared<v1::Reshape>(indices, flat_shape, false);
    const auto scattered =
        std::make_shared<v12::ScatterElementsUpdate>(zeros,
                                                     flat_indices,
                                                     flat_data,
                                                     v0::Constant::create(ov::element::i64, {}, {0}),
                                                     v12::ScatterElementsUpdate::Reduction::NONE);
    return {std::make_shared<v1::Reshape>(scattered, output_shape, false)};
}

ONNX_OP("MaxUnpool", OPSET_SINCE(1), ai_onnx::opset_9::max_unpool);
}  // namespace ov::frontend::onnx::ai_onnx::opset_9
