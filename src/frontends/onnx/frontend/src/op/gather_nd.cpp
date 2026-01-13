// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "openvino/op/gather_nd.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

namespace {
// Helper function to extract a dimension from a shape tensor at given index
ov::Output<ov::Node> get_dimension(const ov::Output<ov::Node>& shape, int64_t index) {
    auto axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto start = v0::Constant::create(ov::element::i64, ov::Shape{1}, {index});
    auto stop = v0::Constant::create(ov::element::i64, ov::Shape{1}, {index + 1});
    auto step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    return std::make_shared<v8::Slice>(shape, start, stop, step, axis);
}
}  // namespace

ov::OutputVector gather_nd(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector ng_inputs{node.get_ov_inputs()};
    auto data = ng_inputs.at(0);
    auto indices = ng_inputs.at(1);
    const auto batch_dims = node.get_attribute_value<int64_t>("batch_dims", 0);

    // If batch_dims > 0, we need to handle broadcasting for batch dimensions
    // This is a workaround for ONNXRuntime's non-standard behavior that allows
    // dimension 1 to broadcast to any size N in batch dimensions
    if (batch_dims > 0) {
        // Check if we can determine statically that broadcasting is not needed
        bool need_broadcast = false;
        bool shapes_are_static = data.get_partial_shape().is_static() && indices.get_partial_shape().is_static();

        if (shapes_are_static) {
            // Compare batch dimensions statically
            auto data_shape_static = data.get_shape();
            auto indices_shape_static = indices.get_shape();

            for (int64_t i = 0; i < batch_dims; ++i) {
                if (data_shape_static[i] != indices_shape_static[i]) {
                    need_broadcast = true;
                    break;
                }
            }
        } else {
            // Dynamic shapes - conservatively assume broadcast may be needed
            need_broadcast = true;
        }

        // Only add Broadcast operations if needed
        if (need_broadcast) {
            auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
            auto indices_shape = std::make_shared<v3::ShapeOf>(indices, ov::element::i64);

            // Compute target batch shape as max(data_batch_shape, indices_batch_shape)
            ov::OutputVector batch_dims_vec;
            for (int64_t i = 0; i < batch_dims; ++i) {
                auto data_dim = get_dimension(data_shape, i);
                auto indices_dim = get_dimension(indices_shape, i);
                auto max_dim = std::make_shared<v1::Maximum>(data_dim, indices_dim);
                batch_dims_vec.push_back(max_dim);
            }

            // Get remaining dimensions
            auto zero_const = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto batch_dims_const = v0::Constant::create(ov::element::i64, ov::Shape{1}, {batch_dims});
            auto one_step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

            auto data_rank_node = std::make_shared<v3::ShapeOf>(data_shape, ov::element::i64);
            auto indices_rank_node = std::make_shared<v3::ShapeOf>(indices_shape, ov::element::i64);

            auto data_remaining =
                std::make_shared<v8::Slice>(data_shape, batch_dims_const, data_rank_node, one_step, zero_const);
            auto indices_remaining =
                std::make_shared<v8::Slice>(indices_shape, batch_dims_const, indices_rank_node, one_step, zero_const);

            // Construct target shapes
            auto target_batch_shape = std::make_shared<v0::Concat>(batch_dims_vec, 0);
            auto target_data_shape =
                std::make_shared<v0::Concat>(ov::OutputVector{target_batch_shape, data_remaining}, 0);
            auto target_indices_shape =
                std::make_shared<v0::Concat>(ov::OutputVector{target_batch_shape, indices_remaining}, 0);

            // Broadcast data and indices to target shapes
            data = std::make_shared<v3::Broadcast>(data, target_data_shape);
            indices = std::make_shared<v3::Broadcast>(indices, target_indices_shape);
        }
    }

    return {std::make_shared<v8::GatherND>(data, indices, batch_dims)};
}

ONNX_OP("GatherND", OPSET_SINCE(1), ai_onnx::opset_1::gather_nd);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
