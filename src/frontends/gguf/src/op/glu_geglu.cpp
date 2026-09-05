// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <utility>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_glu_geglu(const NodeContext& context) {
    auto inputs = get_glu_inputs(context);

    // ggml's GGML_GLU_OP_GEGLU uses the tanh GELU approximation, not OV's default ERF form. The
    // ERF/tanh difference is small per call but compounds across layers into a wrong argmax on
    // deep models (e.g. gemma3-1b), so match ggml with TANH.
    auto gelu = std::make_shared<ov::op::v7::Gelu>(inputs.first, ov::op::GeluApproximationMode::TANH);
    auto res = std::make_shared<ov::op::v1::Multiply>(gelu, inputs.second);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

OutputVector translate_glu_geglu_quick(const NodeContext& context) {
    auto inputs = get_glu_inputs(context);
    auto coefficient = ov::op::v0::Constant::create(inputs.first.get_element_type(), {}, {1.702f});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(inputs.first, coefficient);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(scaled);
    auto gate = std::make_shared<ov::op::v1::Multiply>(inputs.first, sigmoid);
    auto res = std::make_shared<ov::op::v1::Multiply>(gate, inputs.second);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
