// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>
#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_cpy(const NodeContext& context) {
    const int op_case = context.get_op_case();
    const auto input_shape = context.get_input_shape(0);
    const auto output_shape = context.get_output_shape();

    if (op_case == 1 && (context.get_input_size() == 1 || !context.has_input(context.get_input_names()[1]))) {
        ov::Output<ov::Node> value = context.get_input(0);
        // The translated GDN value intentionally has a dynamic packed-row axis, while the CPY's
        // declared input layout may already be reinterpreted as the destination cache. Get the
        // actual packed row width directly from the producing GGML tensor metadata.
        const int64_t packed_width = context.get_attribute<int64_t>("gdn_packed_width", -1);
        FRONT_END_OP_CONVERSION_CHECK(packed_width > 0, "GDN cache copy requires a static packed-row width");
        const int64_t state_rows = static_cast<int64_t>(ov::shape_size(output_shape.to_shape())) / packed_width;
        value = std::make_shared<ov::op::v8::Slice>(value,
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {-state_rows}),
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX}),
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
        if (value.get_element_type() != context.get_output_type()) {
            value = std::make_shared<ov::op::v0::Convert>(value, context.get_output_type());
        }
        auto target = ov::op::v0::Constant::create(ov::element::i64, {output_shape.size()}, output_shape.to_shape());
        auto res = std::make_shared<ov::op::v1::Reshape>(value, target, false);
        return rename_outputs_with_suffix({std::move(res)}, context.get_name());
    }

    if (op_case == 3 && input_shape.is_static() && ov::shape_size(input_shape.to_shape()) == 0) {
        return {context.get_input(1)};
    }

    if (op_case == 4) {
        ov::Output<ov::Node> src = context.get_input(0);
        auto base = context.get_input(1);

        int64_t n_elems = 1;
        for (const auto& dim : input_shape.to_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        const int64_t begin_val = context.get_attribute<int64_t>("cpy_output_offset_elems", 0);
        const int64_t end_val = begin_val + n_elems;
        auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, false);
        if (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        auto head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, 3);
        return rename_outputs_with_suffix({std::move(res)}, context.get_name());
    }

    const std::string writeback_name = context.get_attribute<std::string>("rs_writeback_name", context.get_name());
    const std::string slot_begin_name = "rs_slot_begin_" + writeback_name;
    if (context.has_input(slot_begin_name) && op_case >= 1 && op_case <= 3) {
        const int64_t slot_axis = 2;
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});
        const int64_t feature_width = output_shape[3].get_length();
        auto reshape_writeback = [&](const ov::Output<ov::Node>& value) {
            auto value_shape = std::make_shared<ov::op::v3::ShapeOf>(value, ov::element::i64);
            auto total =
                std::make_shared<ov::op::v1::ReduceProd>(value_shape,
                                                         ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                         true);
            auto rows = std::make_shared<ov::op::v1::Divide>(
                total,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_width}));
            auto target = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 1}),
                                 rows,
                                 ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_width})},
                0);
            return std::make_shared<ov::op::v1::Reshape>(value, target, false);
        };

        ov::Output<ov::Node> src;
        auto begin = context.get_input(slot_begin_name);
        if (op_case == 1) {
            const int64_t packed_width = context.get_attribute<int64_t>("gdn_packed_width", -1);
            FRONT_END_OP_CONVERSION_CHECK(packed_width > 0, "GDN writeback requires a static packed-row width");
            const int64_t state_rows = output_shape[3].get_length() / packed_width;
            auto src_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-state_rows});
            auto state_part = std::make_shared<ov::op::v8::Slice>(context.get_input(0), src_begin, int_max, one, axis);
            src = reshape_writeback(state_part);
            src.get_node_shared_ptr()->set_friendly_name("gdn_writeback_source_" + context.get_name());
        } else if (op_case == 2) {
            const int64_t window_size = input_shape[3].get_length();
            auto src_begin = context.get_input("rs_src_begin_" + writeback_name);
            auto src_end =
                std::make_shared<ov::op::v1::Add>(src_begin,
                                                  ov::op::v0::Constant::create(ov::element::i64, {1}, {window_size}));
            auto window = std::make_shared<ov::op::v8::Slice>(context.get_input(0),
                                                              src_begin,
                                                              src_end,
                                                              one,
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
            src = reshape_writeback(window);
            src.get_node_shared_ptr()->set_friendly_name("conv_writeback_source_" + context.get_name());
        } else {
            src = context.get_input(0);
        }

        if (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        auto base = context.get_input(1);
        auto src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64),
                                                 axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto end = std::make_shared<ov::op::v1::Add>(begin, src_len);
        auto head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, slot_axis);
        return rename_outputs_with_suffix({std::move(res)}, context.get_name());
    }

    ov::Output<ov::Node> res =
        std::make_shared<ov::op::v0::Convert>(context.get_input(0),
                                              context.get_attribute<ov::element::Type>("output_type"));

    // A CPY may reinterpret the source layout into its destination's (e.g. qwen3-next's conv-state
    // writeback flattens the contiguous [S, F] conv_state_last into the flat [S*F] recurrent cache
    // row). When the destination (output) shape differs from the source but holds the same number of
    // elements, reshape to the output layout so the model result matches the ggml cache tensor.
    const auto& out_ps = context.get_output_shape();
    const auto& in_ps = res.get_partial_shape();
    if (out_ps.is_static() && in_ps.is_static() && in_ps != out_ps) {
        const auto in_shape = in_ps.to_shape();
        const auto out_shape = out_ps.to_shape();
        int64_t in_elems = 1, out_elems = 1;
        for (auto d : in_shape)
            in_elems *= static_cast<int64_t>(d);
        for (auto d : out_shape)
            out_elems *= static_cast<int64_t>(d);
        // Only reinterpret when the element counts match. A zero-element CPY (empty defrag/no-op
        // writeback, e.g. a decode step's [.,.,0,.] conv-state copy) keeps its own shape -- reshaping
        // it to the non-empty cache layout would be an invalid element-count change.
        if (in_elems == out_elems && in_elems != 0) {
            std::vector<int64_t> tgt(out_shape.begin(), out_shape.end());
            res =
                std::make_shared<ov::op::v1::Reshape>(res,
                                                      ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt),
                                                      false);
        }
    } else if (out_ps.is_static() && in_ps.rank().is_static() && in_ps != out_ps) {
        // The input carries a (possibly spurious) dynamic axis but the destination cache row is fully
        // static (qwen3-next conv-state writeback: in [1,1,4096,1..3] -> cache [1,1,1,12288]). Reshape
        // to the static cache layout directly: at runtime the dynamic axis takes the value that makes
        // the element counts match, so a plain static target is valid. Guard on the static-dim product
        // dividing the output (a zero-element or incompatible copy keeps its own shape).
        int64_t in_static = 1;
        bool has_dyn = false;
        for (int64_t i = 0; i < in_ps.rank().get_length(); ++i) {
            if (in_ps[i].is_static()) {
                in_static *= in_ps[i].get_length();
            } else {
                has_dyn = true;
            }
        }
        const auto out_shape = out_ps.to_shape();
        int64_t out_elems = 1;
        for (auto d : out_shape)
            out_elems *= static_cast<int64_t>(d);
        // out_elems != 0: an empty destination must not be reshaped (0 % anything == 0).
        if (has_dyn && in_static != 0 && out_elems != 0 && out_elems % in_static == 0) {
            std::vector<int64_t> tgt(out_shape.begin(), out_shape.end());
            res =
                std::make_shared<ov::op::v1::Reshape>(res,
                                                      ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt),
                                                      false);
        }
    }
    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
