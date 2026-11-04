// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/eye.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// Broadcast a vector across the rows of an identity matrix.
OutputVector translate_diag(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto x = context.get_input(0);  // OV shape: [ne3, ne2, 1, ne0]

    auto n = get_dimensions(x, {-1});
    auto zero = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto eye = std::make_shared<ov::op::v9::Eye>(n, n, zero, x.get_element_type());

    auto res = std::make_shared<ov::op::v1::Multiply>(x, eye);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
