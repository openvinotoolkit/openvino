// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/tile.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape; every destination dimension is an
// integer multiple of the corresponding source dimension.
OutputVector translate_repeat(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    auto input = context.get_input(0);
    const auto repeats_attribute = context.get_attribute_as_any("repeats");
    ov::Output<ov::Node> factors;
    if (!repeats_attribute.empty()) {
        const auto repeats = repeats_attribute.as<std::vector<int64_t>>();
        FRONT_END_OP_CONVERSION_CHECK(!repeats.empty(), "REPEAT requires non-empty 'repeats'");
        factors = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
    } else {
        FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() == 2,
                                      "REPEAT requires 'repeats' or a target-shape operand");
        auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
        auto target_shape = std::make_shared<ov::op::v3::ShapeOf>(context.get_input(1), ov::element::i64);
        factors = std::make_shared<ov::op::v1::Divide>(target_shape, input_shape);
    }
    return rename_outputs_with_suffix({std::make_shared<ov::op::v0::Tile>(input, factors)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
