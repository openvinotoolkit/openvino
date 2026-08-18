// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/topk.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// ggml_top_k(a, k): the indices of the k largest values along ne[0] (the OV last axis),
// ordered by descending value, as i32. k is the extent of that axis on the output.
OutputVector translate_top_k(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    const int64_t k = context.get_output_shape()[context.get_output_shape().size() - 1].get_length();
    auto k_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {k});
    auto topk = std::make_shared<ov::op::v11::TopK>(input,
                                                    k_node,
                                                    -1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    context.get_attribute<ov::element::Type>("output_type"));

    return rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
