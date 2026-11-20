// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_ARANGE has fixed start/stop/step parameters, so its values are a Constant.
OutputVector translate_arange(const NodeContext& context) {
    num_inputs_check(context, 0, 0);

    const auto start = context.get_attribute<float>("start");
    const auto step = context.get_attribute<float>("step");
    const auto shape = context.get_output_shape();
    FRONT_END_OP_CONVERSION_CHECK(shape.is_static(), "ARANGE requires a static output shape");

    const auto out_shape = shape.to_shape();
    std::vector<float> values(ov::shape_size(out_shape));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = start + static_cast<float>(i) * step;
    }
    auto res = ov::op::v0::Constant::create(ov::element::f32, out_shape, values);
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
