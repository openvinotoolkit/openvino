// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/convert.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector tile(const ov::frontend::onnx::Node& node) {
    auto input = node.get_ov_inputs().at(0);
    auto repeats = node.get_ov_inputs().at(1);

    const auto& in_pshape = input.get_partial_shape();
    const bool is_scalar_input = in_pshape.rank().is_static() && in_pshape.rank().get_length() == 0;

    const auto& repeats_pshape = repeats.get_partial_shape();
    const bool is_empty_1d_repeats =
        common::is_failsafe_node(repeats.get_node_shared_ptr()) ||
        (repeats_pshape.is_static() && repeats_pshape.rank().get_length() == 1 && repeats_pshape[0] == 0);

    // In ONNX, tiling a rank-0 scalar with length-0 (1-D) repeats is an identity operation.
    // Length-0 repeats are converted to failsafe nodes or empty 1-D constants by the frontend.
    if (is_scalar_input && is_empty_1d_repeats) {
        common::mark_as_optimized_out(input);
        return {input};
    }

    // Workaround for backends which require repeats to be i64.
    // Remove the following line when no longer needed.
    repeats = std::make_shared<v0::Convert>(repeats, ov::element::i64);

    return {std::make_shared<v0::Tile>(input, repeats)};
}

ONNX_OP("Tile", OPSET_SINCE(1), ai_onnx::opset_1::tile);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
