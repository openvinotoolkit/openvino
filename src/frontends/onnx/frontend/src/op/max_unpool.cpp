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
#include "openvino/op/variadic_split.hpp"
#include "openvino/util/common_util.hpp"
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

    // The bound keeps 'k - s - pb - pe' far from int64 overflow
    static constexpr int64_t max_value = std::numeric_limits<int32_t>::max();
    const auto is_positive = [](int64_t v) {
        return v > 0 && v <= max_value;
    };
    const auto is_non_negative = [](int64_t v) {
        return v >= 0 && v <= max_value;
    };
    CHECK_VALID_NODE(node,
                     spatial_rank > 0 && std::all_of(kernel_shape.begin(), kernel_shape.end(), is_positive),
                     "MaxUnpool 'kernel_shape' attribute must be non-empty and positive (<= INT32_MAX). Got: ",
                     ov::util::vector_to_string(kernel_shape));
    CHECK_VALID_NODE(node,
                     strides.size() == spatial_rank && std::all_of(strides.begin(), strides.end(), is_positive),
                     "MaxUnpool 'strides' attribute must have ",
                     spatial_rank,
                     " positive (<= INT32_MAX) elements. Got: ",
                     ov::util::vector_to_string(strides));
    CHECK_VALID_NODE(node,
                     pads.size() == spatial_rank * 2 && std::all_of(pads.begin(), pads.end(), is_non_negative),
                     "MaxUnpool 'pads' attribute must have ",
                     spatial_rank * 2,
                     " non-negative (<= INT32_MAX) elements. Got: ",
                     ov::util::vector_to_string(pads));
    const auto data_rank = data.get_partial_shape().rank();
    CHECK_VALID_NODE(node,
                     data_rank.is_dynamic() || data_rank.get_length() == static_cast<int64_t>(spatial_rank + 2),
                     "MaxUnpool input rank must be 'kernel_shape' size + 2. Got rank: ",
                     data_rank,
                     ", 'kernel_shape' size: ",
                     spatial_rank);

    CHECK_VALID_NODE(node,
                     data.get_partial_shape().compatible(indices.get_partial_shape()),
                     "MaxUnpool 'indices' shape must match the input shape. Got: ",
                     indices.get_partial_shape(),
                     " and ",
                     data.get_partial_shape());

    ov::Output<ov::Node> output_shape;
    if (common::is_input_valid(node, 2)) {
        output_shape = inputs[2];
        // A runtime 'output_shape' can only be checked by its shape
        const auto& os_shape = output_shape.get_partial_shape();
        CHECK_VALID_NODE(
            node,
            os_shape.rank().is_dynamic() ||
                (os_shape.rank().get_length() == 1 &&
                 (os_shape[0].is_dynamic() || os_shape[0].get_length() == static_cast<int64_t>(spatial_rank + 2))),
            "MaxUnpool 'output_shape' must be a 1D tensor with ",
            spatial_rank + 2,
            " elements. Got shape: ",
            os_shape);
        if (const auto os_const = ov::as_type_ptr<v0::Constant>(output_shape.get_node_shared_ptr())) {
            const auto values = os_const->cast_vector<int64_t>();
            const auto& data_shape = data.get_partial_shape();
            CHECK_VALID_NODE(node,
                             std::all_of(values.begin(),
                                         values.end(),
                                         [](int64_t v) {
                                             return v >= 0;
                                         }),
                             "MaxUnpool 'output_shape' must be non-negative. Got: ",
                             ov::util::vector_to_string(values));
            for (size_t i = 0; i < 2 && data_shape.rank().is_static(); ++i) {
                CHECK_VALID_NODE(node,
                                 data_shape[i].is_dynamic() || data_shape[i].get_length() == values[i],
                                 "MaxUnpool 'output_shape' batch and channel dimensions must match the input. Got: ",
                                 ov::util::vector_to_string(values),
                                 " for input shape ",
                                 data_shape);
            }
        }
    } else {
        // out[i] = (in[i] - 1) * strides[i] + kernel_shape[i] - pads_begin[i] - pads_end[i]
        std::vector<int64_t> shift(spatial_rank);
        for (size_t i = 0; i < spatial_rank; ++i) {
            shift[i] = kernel_shape[i] - strides[i] - pads[i] - pads[i + spatial_rank];
        }
        const auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
        // [N, C] and spatial dims
        const auto split = std::make_shared<v1::VariadicSplit>(data_shape,
                                                               v0::Constant::create(ov::element::i64, {}, {0}),
                                                               v0::Constant::create(ov::element::i64, {2}, {2, -1}));
        const auto scaled =
            std::make_shared<v1::Multiply>(split->output(1),
                                           v0::Constant::create(ov::element::i64, {spatial_rank}, strides));
        const auto out_spatial =
            std::make_shared<v1::Add>(scaled, v0::Constant::create(ov::element::i64, {spatial_rank}, shift));
        output_shape = std::make_shared<v0::Concat>(ov::OutputVector{split->output(0), out_spatial}, 0);
    }

    // Indices address flat(N x C x D1 x ... x Dn)
    const auto zero = std::make_shared<v1::ConvertLike>(v0::Constant::create(ov::element::f32, {}, {0}), data);
    const auto total_size =
        std::make_shared<v1::ReduceProd>(output_shape, v0::Constant::create(ov::element::i64, {1}, {0}), true);
    const auto zeros = std::make_shared<v3::Broadcast>(zero, total_size);
    const auto flat_shape = v0::Constant::create(ov::element::i64, {1}, {-1});
    const auto flat_data = std::make_shared<v1::Reshape>(data, flat_shape, false);
    const auto flat_indices = std::make_shared<v1::Reshape>(indices, flat_shape, false);
    const auto scattered = std::make_shared<v12::ScatterElementsUpdate>(zeros,
                                                                        flat_indices,
                                                                        flat_data,
                                                                        v0::Constant::create(ov::element::i64, {}, {0}),
                                                                        v12::ScatterElementsUpdate::Reduction::NONE);
    return {std::make_shared<v1::Reshape>(scattered, output_shape, false)};
}

ONNX_OP("MaxUnpool", OPSET_SINCE(1), ai_onnx::opset_9::max_unpool);
}  // namespace ov::frontend::onnx::ai_onnx::opset_9
