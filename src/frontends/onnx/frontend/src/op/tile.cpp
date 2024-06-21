// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/convert.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector tile(const ov::frontend::onnx::Node& node) {
    auto input = node.get_ov_inputs().at(0);
    auto repeats = node.get_ov_inputs().at(1);

    // Workaround for backends which require repeats to be i64.
    // Remove the following line when no longer needed.
    repeats = std::make_shared<v0::Convert>(repeats, ov::element::i64);

    return {std::make_shared<v0::Tile>(input, repeats)};
}

static bool registered = register_translator("Tile", VersionRange::single_version_for_all_opsets(), tile);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
