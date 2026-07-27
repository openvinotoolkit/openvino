// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_SET writes src[1] into a contiguous region of src[0] and returns the updated tensor.
// ggml stores the destination byte offset in op_params[3]; the decoder converts it to an element
// count and exposes it as the "set_offset_elems" attribute, so the frontend never touches raw
// strides. Both inputs arrive already resolved (translate_view materialized any views), so we flatten
// dst, scatter the flattened src into [offset, offset + numel(src)), and reshape back to dst's shape.
OutputVector translate_set(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto dst = context.get_input(0);
    auto src = context.get_input(1);

    src = std::make_shared<ov::op::v0::Convert>(src, context.get_attribute<ov::element::Type>("output_type"));

    const int64_t offset_elems = context.get_attribute<int64_t>("set_offset_elems");

    auto dst_flat = std::make_shared<ov::op::v1::Reshape>(
        dst, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
    auto src_flat = std::make_shared<ov::op::v1::Reshape>(
        src, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);

    // Indices [offset, offset + numel(src)) into the flattened destination.
    auto src_shape = std::make_shared<ov::op::v3::ShapeOf>(src_flat, ov::element::i64);
    auto src_len = std::make_shared<ov::op::v1::ReduceProd>(
        src_shape, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}), false);
    auto start = ov::op::v0::Constant::create(ov::element::i64, {}, {offset_elems});
    auto stop = std::make_shared<ov::op::v1::Add>(start, src_len);
    auto step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto indices = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::i64);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});

    auto updated_flat = std::make_shared<ov::op::v3::ScatterUpdate>(dst_flat, indices, src_flat, axis);

    auto dst_shape = std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64);
    auto res = std::make_shared<ov::op::v1::Reshape>(updated_flat, dst_shape, false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
