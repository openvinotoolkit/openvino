// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_get_rows(const NodeContext& context) {
    const bool relative_shift = context.get_attribute<bool>("relative_shift", false);
    num_inputs_check(context, relative_shift ? 1 : 2, relative_shift ? 1 : 2);

    Output<Node> res;
    auto data = context.get_input(0);
    ov::Output<ov::Node> indices;
    if (!relative_shift)
        indices = context.get_input(1);
    // GGML gathers integer rows as I32 and all floating/quantized rows as F32.
    const auto type = data.get_element_type() == ov::element::i32 ? ov::element::i32 : ov::element::f32;
    const auto convert_rows = [&](ov::Output<Node> rows) {
        if (rows.get_element_type() != type) {
            rows = std::make_shared<ov::op::v0::Convert>(rows, type);
        }
        return rows;
    };

    if (relative_shift) {
        const auto scalar = [](int64_t value) {
            return ov::op::v0::Constant::create(ov::element::i64, {}, {value});
        };
        auto length = std::make_shared<ov::op::v0::Squeeze>(get_dimensions(data, {2}));
        auto range = std::make_shared<ov::op::v4::Range>(scalar(0), length, scalar(1), ov::element::i64);
        auto row =
            std::make_shared<ov::op::v0::Unsqueeze>(range,
                                                    ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 3}));
        auto column =
            std::make_shared<ov::op::v0::Unsqueeze>(range,
                                                    ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        indices = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Subtract>(length, scalar(1)), row),
            column);
    }
    if (context.get_attribute<bool>("moe_weight_gather", false)) {
        const auto ids_shape = indices.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(
            ids_shape.rank().is_static() && ids_shape.rank().get_length() == 4 && ids_shape[3].is_static(),
            "MoE weight gather expects rank-4 indices with a static selected-expert dimension");
        const auto selected_experts = ids_shape[3].get_length();
        auto idx = std::make_shared<ov::op::v0::Convert>(indices, ov::element::i32);
        ov::Output<Node> gathered;
        if (context.get_attribute<bool>("shared_row", false)) {
            auto row = std::make_shared<ov::op::v1::Reshape>(data,
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
                                                             false);
            gathered =
                std::make_shared<ov::op::v8::Gather>(row, idx, ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        } else {
            gathered = std::make_shared<ov::op::v6::GatherElements>(data, idx, -1);
        }
        auto columns = std::make_shared<ov::op::v1::Reshape>(
            gathered,
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, selected_experts, 1}),
            false);
        return rename_outputs_with_suffix({convert_rows(columns)}, context.get_name());
    }
    if (context.get_attribute<bool>("gather_row_elements", false) || relative_shift) {
        // Select a column independently in each row; singleton index batches broadcast.
        auto shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{get_dimensions(data, {0, 1, 2}), get_dimensions(indices, {3})},
            0);
        indices = std::make_shared<ov::op::v3::Broadcast>(indices, shape);
        return rename_outputs_with_suffix(
            {convert_rows(std::make_shared<ov::op::v6::GatherElements>(data, indices, 3))},
            context.get_name());
    }
    if (context.get_attribute<bool>("data_only", false)) {
        return {convert_rows(data)};
    }

    // Rows gathered independently in every [ne3, ne2] batch: data [ne3, ne2, ne1, ne0], indices [1, ne3, ne2, k].
    if (context.get_attribute<bool>("gather_rows", false)) {
        auto batched =
            std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        res = std::make_shared<ov::op::v8::Gather>(data,
                                                   batched,
                                                   ov::op::v0::Constant::create(ov::element::i64, {}, {2}),
                                                   2);
        return rename_outputs_with_suffix({convert_rows(res)}, context.get_name());
    }

    if (context.get_attribute<bool>("indices_view", false)) {
        // The input comes from a VIEW
        indices = process_view_input(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    indices =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
    if (data.get_partial_shape().rank() == 4) {
        if (!(data.get_partial_shape()[1].is_dynamic()) && data.get_partial_shape()[1].get_length() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            data = std::make_shared<ov::op::v0::Squeeze>(data,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        } else {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
        }
    } else {
        res = std::make_shared<ov::op::v8::Gather>(data,
                                                   indices,
                                                   ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
    }

    // The two Squeezes above dropped the leading axes; restore ggml's rank-4 form.
    res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    res = apply_hadamard(context, res);
    return rename_outputs_with_suffix({convert_rows(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
