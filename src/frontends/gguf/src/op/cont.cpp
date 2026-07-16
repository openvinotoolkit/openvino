// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_cont(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_attribute<int>("op_case", 0);
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported CONT case");

    auto dst_shape = context.get_output_shape().to_shape();
    ov::Output<Node> res;

    if (op_case == 1) {
        // The input comes from a PERMUTE
        throw std::runtime_error("Code of this case might be outdated");
        dst_shape[1] = -1;
        res = std::make_shared<ov::op::v1::Reshape>(
            context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), false);
    } else if (op_case == 2) {
        // The input comes from a TRANSPOSE
        return {context.get_input(0)};
    } else {
        // The input comes from a VIEW
        res = process_view_input(context, 0);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
