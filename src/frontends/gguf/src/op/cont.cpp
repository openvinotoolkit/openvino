// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_CONT materializes its operand in ggml's contiguous layout. The operand is already the logical
// tensor (any PERMUTE/TRANSPOSE/VIEW was translated by its own op), so CONT only applies the dimension
// merge/split ggml_cont_* may perform. The decoder describes it as "cont_reshape": the output dims in
// OpenVINO order with -1 on the runtime (token) axis.
OutputVector translate_cont(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto target = context.get_attribute<std::vector<int64_t>>("cont_reshape", {});
    if (target.empty()) {
        // No target means pass-through, which is only valid if the operand already has the CONT's layout: every
        // dimension static in the operand is unchanged (a runtime axis may have been captured with a value).
        const auto output_shape = context.get_output_shape();
        const auto input_shape = input.get_partial_shape();
        bool same_layout = output_shape.rank().is_dynamic();
        if (!same_layout && input_shape.rank().is_static() && input_shape.rank() == output_shape.rank()) {
            same_layout = true;
            for (int64_t i = 0; i < input_shape.rank().get_length(); ++i) {
                same_layout = same_layout &&
                              (input_shape[i].is_dynamic() || (output_shape[i].is_static() && output_shape[i] == input_shape[i]));
            }
        }
        FRONT_END_OP_CONVERSION_CHECK(same_layout,
                                      "CONT that changes the shape requires a \"cont_reshape\" target: input ",
                                      input.get_partial_shape(),
                                      " != output ",
                                      output_shape);
        // A pass-through CONT must not rename the shared input producer.
        return {input};
    }
    auto pattern = ov::op::v0::Constant::create(ov::element::i64, {target.size()}, target);
    auto res = std::make_shared<ov::op::v1::Reshape>(input, pattern, false);
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

// GGML_OP_DUP copies its operand into a tensor of the output's type. The copy is CONT's; a differing output type
// ("dst_type", the output's element type) is a conversion on top of it.
OutputVector translate_dup(const NodeContext& context) {
    auto outputs = translate_cont(context);
    const auto dst_type = context.get_attribute<ov::element::Type>("dst_type", ov::element::dynamic);
    if (dst_type.is_static() && outputs[0].get_element_type() != dst_type) {
        auto res = std::make_shared<ov::op::v0::Convert>(outputs[0], dst_type);
        return rename_outputs_with_suffix({std::move(res)}, context.get_name());
    }
    return outputs;
}

}  // namespace ov::frontend::gguf::op
