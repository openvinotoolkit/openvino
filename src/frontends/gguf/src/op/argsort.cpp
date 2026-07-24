// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_ARGSORT: return the indices that sort the last dimension. The decoder maps ggml's
// GGML_SORT_ORDER_ASC/DESC enum to a plain int "sort_order" (0 = ascending, 1 = descending), so
// the translator needs no ggml headers. Implemented via TopK over the full last dim (k == dim).
OutputVector translate_argsort(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    const int sort_order = context.get_attribute<int>("sort_order");

    ov::op::v11::TopK::Mode mode;
    switch (sort_order) {
    case 0:
        mode = ov::op::v11::TopK::Mode::MIN;
        break;
    case 1:
        mode = ov::op::v11::TopK::Mode::MAX;
        break;
    default:
        FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported ARGSORT order: ", sort_order);
    }

    auto index_type = context.get_attribute<ov::element::Type>("output_type");
    // ggml ARGSORT sorts ne[0] == the OV last axis; derive it from the rank (rank-4 keeps axis 3).
    const auto& in_ps = input.get_partial_shape();
    const int64_t axis = in_ps.rank().is_static() ? in_ps.rank().get_length() - 1 : 3;
    auto k = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(input.get_node_shared_ptr(), {(int)axis}),
                                                   ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    auto topk = std::make_shared<ov::op::v11::TopK>(input,
                                                    k,
                                                    axis,
                                                    mode,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    index_type,
                                                    false);

    return rename_outputs_with_suffix({topk->output(1)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
