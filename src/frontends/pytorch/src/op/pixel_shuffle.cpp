// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/split.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

// Holds the split shape information; dims_before captures all pre-spatial dims
struct PixelSpatialInfo {
    Output<Node> dims_before;
    Output<Node> channels;
    Output<Node> height;
    Output<Node> width;
};

PixelSpatialInfo get_pixel_spatial_info(const NodeContext& context, const Output<Node>& x) {
    auto zero_vec = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto zero_scalar = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto neg_three = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-3}));
    auto one_vec = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i32));
    auto dims_before = context.mark_node(std::make_shared<v8::Slice>(shape, zero_vec, neg_three, one_vec));
    auto indices = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {-3, -2, -1}));
    auto dims = context.mark_node(std::make_shared<v8::Gather>(shape, indices, zero_scalar));
    auto split = context.mark_node(std::make_shared<v1::Split>(dims, zero_scalar, 3));
    return {dims_before, split->output(0), split->output(1), split->output(2)};
}

Output<Node> make_flatten_shape(const NodeContext& context,
                                const Output<Node>& channels,
                                const Output<Node>& height,
                                const Output<Node>& width) {
    auto neg_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto chw = context.mark_node(std::make_shared<v0::Concat>(OutputVector{channels, height, width}, 0));
    return context.mark_node(std::make_shared<v0::Concat>(OutputVector{neg_one, chw}, 0));
}

Output<Node> make_final_shape(const NodeContext& context,
                              const Output<Node>& dims_before,
                              const Output<Node>& new_c,
                              const Output<Node>& new_h,
                              const Output<Node>& new_w) {
    auto tail = context.mark_node(std::make_shared<v0::Concat>(OutputVector{new_c, new_h, new_w}, 0));
    return context.mark_node(std::make_shared<v0::Concat>(OutputVector{dims_before, tail}, 0));
}

OutputVector translate_pixel_transform(const NodeContext& context, bool is_shuffle) {
    num_inputs_check(context, 2, 2);
    const auto x = context.get_input(0);
    const auto block = context.const_input<int64_t>(1);
    PYTORCH_OP_CONVERSION_CHECK(block > 0, "Upscale factor for pixel shuffle ops must be positive");

    const auto block_size = static_cast<size_t>(block);
    const auto block_scalar =
        context.mark_node(v0::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(block)}));
    const auto block_sq_scalar =
        context.mark_node(v0::Constant::create(element::i32, Shape{}, {static_cast<int32_t>(block * block)}));

    const auto [dims_before, channels, height, width] = get_pixel_spatial_info(context, x);
    const auto flat_shape = make_flatten_shape(context, channels, height, width);
    const auto flattened = context.mark_node(std::make_shared<v1::Reshape>(x, flat_shape, false));

    Output<Node> transformed;
    Output<Node> new_c;
    Output<Node> new_h;
    Output<Node> new_w;

    if (is_shuffle) {
        transformed = context.mark_node(
            std::make_shared<v0::DepthToSpace>(flattened, v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, block_size));
        new_c = context.mark_node(std::make_shared<v1::Divide>(channels, block_sq_scalar));
        new_h = context.mark_node(std::make_shared<v1::Multiply>(height, block_scalar));
        new_w = context.mark_node(std::make_shared<v1::Multiply>(width, block_scalar));
    } else {
        transformed = context.mark_node(
            std::make_shared<v0::SpaceToDepth>(flattened, v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, block_size));
        new_c = context.mark_node(std::make_shared<v1::Multiply>(channels, block_sq_scalar));
        new_h = context.mark_node(std::make_shared<v1::Divide>(height, block_scalar, true));
        new_w = context.mark_node(std::make_shared<v1::Divide>(width, block_scalar, true));
    }

    const auto final_shape = make_final_shape(context, dims_before, new_c, new_h, new_w);
    auto reshaped = context.mark_node(std::make_shared<v1::Reshape>(transformed, final_shape, false));
    return {std::move(reshaped)};
}

}  // namespace

OutputVector translate_pixel_shuffle(const NodeContext& context) {
    // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
    return translate_pixel_transform(context, true);
};

OutputVector translate_pixel_unshuffle(const NodeContext& context) {
    // aten::pixel_unshuffle(Tensor self, int upscale_factor) -> Tensor
    return translate_pixel_transform(context, false);
};

OutputVector translate_channel_shuffle(const NodeContext& context) {
    // aten::channel_shuffle(Tensor self, int groups) -> Tensor
    num_inputs_check(context, 2, 2);
    const auto x = context.get_input(0);
    const auto groups = context.const_input<int64_t>(1);
    PYTORCH_OP_CONVERSION_CHECK(groups > 0, "groups argument for channel_shuffle must be positive");
    auto shuffled = context.mark_node(std::make_shared<v0::ShuffleChannels>(x, 1, static_cast<size_t>(groups)));
    return {std::move(shuffled)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov