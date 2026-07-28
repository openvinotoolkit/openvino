// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs expand_v2(const NodeContext& node) {
    using namespace default_opset;
    auto x = node.get_input("X");
    Output<Node> shape_expected_node;
    if (node.has_input("Shape")) {
        shape_expected_node = node.get_input("Shape");
    } else if (node.has_input("expand_shapes_tensor")) {
        auto inputs = node.get_ng_inputs("expand_shapes_tensor");
        ov::NodeVector node_vec;
        for (auto& input : inputs) {
            if (input.get_partial_shape().rank().get_length() == 0) {
                // should unsqueeze the input with non-shape.
                auto unsqueeze_scalar = default_opset::Constant::create(ov::element::i32, {}, {0});
                input = std::make_shared<default_opset::Unsqueeze>(input, unsqueeze_scalar);
            }
            PADDLE_OP_CHECK(node,
                            input.get_partial_shape().rank().get_length() == 1,
                            "the rank of conv input must == 1");
            auto cast = std::make_shared<Convert>(input, element::i32);
            node_vec.emplace_back(cast);
        }
        shape_expected_node = std::make_shared<Concat>(node_vec, 0);
    } else {
        std::vector<int32_t> shape_expected;
        if (node.has_attribute("shape")) {
            shape_expected = node.get_attribute<std::vector<int32_t>>("shape");
        } else {
            throw std::runtime_error("expand: has no shape attribute");
        }
        shape_expected_node = Constant::create(element::i32, {shape_expected.size()}, shape_expected);
    }
    // expected shape rank
    const auto shape_expected_node_rank = std::make_shared<ShapeOf>(shape_expected_node, element::i32);
    // input shape rank
    const auto input_shape_node_shape = std::make_shared<ShapeOf>(x, element::i32);
    const auto input_shape_node_rank = std::make_shared<ShapeOf>(input_shape_node_shape, element::i32);
    // rank difference
    const auto rank_diff = std::make_shared<Subtract>(shape_expected_node_rank, input_shape_node_rank);
    // axis index needed to add
    const auto rank_idx = std::make_shared<Broadcast>(Constant::create(element::i32, {1}, {1}), rank_diff);
    // add axis
    const auto fixed_input_shape_node = std::make_shared<Concat>(NodeVector{rank_idx, input_shape_node_shape}, 0);

    // if -1 in shape we will copy the original value from input
    auto zero_node = Constant::create(ov::element::i32, {1}, {0});
    auto mask_node = std::make_shared<Greater>(shape_expected_node, zero_node);
    Output<Node> fixed_shape_node = std::make_shared<Select>(mask_node, shape_expected_node, fixed_input_shape_node);

    // The target shape vector can end up with a dynamic length (e.g. when it is assembled from a
    // dynamic-length Slice inside a control-flow body). That would make the Broadcast output rank
    // dynamic, which some plugins (e.g. CPU) do not support. The expand output rank is statically
    // known from the Paddle op output var descriptor, so pin the target shape length to that static
    // rank to keep the Broadcast output rank static without constraining the (possibly dynamic) dims.
    // A Gather with constant indices [0, out_rank) is used (rather than a Reshape) so the static
    // length survives offline shape-of subgraph simplification / nop elimination: the gathered axis
    // length is dynamic, so GatherNopElimination cannot fold it away.
    const auto output_info = node.get_output_port_infos("Out");
    if (!output_info.empty() && output_info[0].second.rank().is_static()) {
        const auto out_rank = static_cast<size_t>(output_info[0].second.rank().get_length());
        std::vector<int32_t> indices(out_rank);
        std::iota(indices.begin(), indices.end(), 0);
        const auto gather_indices = Constant::create(element::i32, {out_rank}, indices);
        const auto gather_axis = Constant::create(element::i32, {}, {0});
        fixed_shape_node = std::make_shared<Gather>(fixed_shape_node, gather_indices, gather_axis);
    }
    return node.default_single_output_mapping({std::make_shared<Broadcast>(x, fixed_shape_node)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov