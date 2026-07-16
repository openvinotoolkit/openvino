// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

#include <memory>
#include <openvino/op/convert.hpp>

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    auto res = std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_attribute<ov::element::Type>("output_type"));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
