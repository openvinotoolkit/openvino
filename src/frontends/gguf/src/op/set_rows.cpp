// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "utils.hpp"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/squeeze.hpp>

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// Always emit the internal SetRows placeholder op (never a device-specific ScatterUpdate). A
// normalization-stage lowering replaces it: the built-in LowerSetRows rebuilds the stateless
// ScatterUpdate form; a caller-registered stateful lowering may instead turn the SetRows that
// feeds attention into a stateful KV-cache subgraph. This keeps conversion identical regardless
// of execution mode and needs no KV-vs-non-KV classification at translate time.
OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    auto dst = context.get_input(2);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_attribute<ov::element::Type>("output_type"));

    auto dst_shape = context.get_output_shape().to_shape();

    auto ind_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
    auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
        data,
        ov::op::v0::Constant::create(ov::element::i64, {4},
                                     {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) dst_shape[3]}),
        false);

    auto set_rows = std::make_shared<SetRows>(data_reshaped, ind_squeezed, dst);
    return rename_outputs_with_suffix({set_rows}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
