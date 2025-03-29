// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_space_to_batch.hpp"

#include <climits>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace std;
using namespace ov::element;

void ov::pass::ConvertSpaceToBatch::convert_space_to_batch() {
    MATCHER_SCOPE(ConvertSpaceToBatch_convert_space_to_batch);
    const auto space_to_batch = pattern::wrap_type<ov::op::v1::SpaceToBatch>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto space_to_batch = ov::as_type_ptr<ov::op::v1::SpaceToBatch>(m.get_match_root());
        if (!space_to_batch || transformation_callback(space_to_batch)) {
            return false;
        }

        const auto data = space_to_batch->input_value(0);
        if (data.get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        const auto block = space_to_batch->input_value(1);
        const auto pads_begin = space_to_batch->input_value(2);
        const auto pads_end = space_to_batch->input_value(3);

        if (block.get_partial_shape().is_dynamic() || block.get_shape().size() == 0) {
            return false;
        }
        const auto block_length = static_cast<int64_t>(block.get_shape()[0]);

        NodeRegistry rg;

        //    Zero-pad the start and end of dimensions [D_0, ..., D_{N - 1}] of the input according to
        //    `pads_begin`
        //    and `pads_end`:
        //    note: P_0 for batch dimension is expected to be 0 (no-padding).
        //      x = [batch + P_0, D_1 + P_1, D_2 + P_2, ..., D_{N - 1} + P_{N - 1}], where P_i =
        //      pads_begin[i] + pads_end[i]
        shared_ptr<Node> flat_node = rg.make<ov::op::v1::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        const auto out_shape = rg.make<ov::op::v3::ShapeOf>(flat_node, block.get_element_type());

        const auto zero = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 0);
        const auto one = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 1);
        const auto int_max = rg.make<ov::op::v0::Constant>(i64, Shape{1}, INT_MAX);

        // First we have to disperse the data from spatial dimensions, then
        // rearrange them so as appropriate chunks of data where close to their
        // destination place. Finally squeeze data from respective dimensions.

        //    note: B_0 for batch is ignored.
        //      x' = reshape(x, [batch, (D_1 + P_1) / B_1, B_1, (D_2 + P_2) / B_2, B_2, ...,
        //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}, B_{N - 1}]), where B_i = block_shape[i]
        const auto batch = rg.make<ov::op::v8::Gather>(out_shape, zero, zero);
        const auto out_shape_tail = rg.make<ov::op::v8::Slice>(out_shape, one, int_max, one);
        const auto block_tail = rg.make<ov::op::v8::Slice>(block, one, int_max, one);
        const auto os_tail_div = rg.make<ov::op::v1::Divide>(out_shape_tail, block_tail);

        // interleave os_tail_div with block_tail
        const auto c = rg.make<ov::op::v0::Concat>(NodeVector{os_tail_div, block_tail}, 0);
        const auto r = rg.make<ov::op::v1::Reshape>(
            c,
            rg.make<ov::op::v0::Constant>(i64, Shape{2}, vector<int64_t>{2, block_length - 1}),
            false);
        const auto t =
            rg.make<ov::op::v1::Transpose>(r, rg.make<ov::op::v0::Constant>(i64, Shape{2}, vector<int64_t>{1, 0}));
        const auto interleaved =
            rg.make<ov::op::v1::Reshape>(t,
                                         rg.make<ov::op::v0::Constant>(i64, Shape{1}, 2 * (block_length - 1)),
                                         false);

        const auto dispersed_shape = rg.make<ov::op::v0::Concat>(NodeVector{batch, interleaved}, 0);
        flat_node = rg.make<ov::op::v1::Reshape>(flat_node, dispersed_shape, false);

        //    x'' = transpose(x',  [2, 4, ..., (N - 1) + (N - 1), 0, 1, 3, ..., N + (N - 1)])
        vector<int64_t> axes_order;
        for (int64_t i = 0, j = 2; i < block_length - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }
        axes_order.push_back(0);
        for (int64_t i = 0, j = 1; i < block_length - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }

        const auto axes_order_const = rg.make<ov::op::v0::Constant>(i64, Shape{axes_order.size()}, axes_order);
        flat_node = rg.make<ov::op::v1::Transpose>(flat_node, axes_order_const);

        //    y = reshape(x'', [batch * B_1 * ... * B_{N - 1}, (D_1 + P_1) / B_1, (D_2 + P_2) / B_2, ...,
        //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}])
        //    note: B_0 is assumed to be 1 by op definion
        const auto block_prod = rg.make<ov::op::v1::ReduceProd>(block, zero);
        const auto squeezed_shape =
            rg.make<ov::op::v0::Concat>(NodeVector{rg.make<ov::op::v1::Multiply>(batch, block_prod), os_tail_div}, 0);
        flat_node = rg.make<ov::op::v1::Reshape>(flat_node, squeezed_shape, false);

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        copy_runtime_info(space_to_batch, rg.get());
        replace_node(space_to_batch, flat_node);
        return true;
    };

    const auto m = make_shared<pattern::Matcher>(space_to_batch, matcher_name);
    this->register_matcher(m, callback);
}

void ov::pass::ConvertSpaceToBatch::convert_space_to_batch_by_elements() {
    MATCHER_SCOPE(ConvertSpaceToBatch_convert_space_to_batch_by_elements);
    const auto space_to_batch = pattern::wrap_type<ov::op::v1::SpaceToBatch>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto space_to_batch = ov::as_type_ptr<ov::op::v1::SpaceToBatch>(m.get_match_root());
        if (!space_to_batch || transformation_callback(space_to_batch)) {
            return false;
        }

        const auto data = space_to_batch->input_value(0);
        if (data.get_partial_shape().rank().is_dynamic()) {
            return false;
        }

        const auto block = space_to_batch->input_value(1);
        const auto pads_begin = space_to_batch->input_value(2);
        const auto pads_end = space_to_batch->input_value(3);

        if (block.get_partial_shape().is_dynamic() || block.get_shape().size() == 0) {
            return false;
        }
        const auto block_length = static_cast<int64_t>(block.get_shape()[0]);

        NodeRegistry rg;

        shared_ptr<Node> flat_node = rg.make<ov::op::v1::Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        shared_ptr<Node> squeezed_shape = rg.make<ov::op::v3::ShapeOf>(flat_node, block.get_element_type());

        const auto zero = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 0);
        const auto one = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 1);
        const auto int_max = rg.make<ov::op::v0::Constant>(i64, Shape{1}, INT_MAX);

        for (int64_t b_idx = block_length - 1; b_idx >= 0; --b_idx) {
            const auto block_index = rg.make<ov::op::v0::Constant>(i64, Shape{1}, b_idx);
            const auto block_index_next = rg.make<ov::op::v0::Constant>(i64, Shape{1}, b_idx + 1);
            const auto block_value = rg.make<ov::op::v8::Gather>(block, block_index, zero);

            NodeVector dispersed_shape_prep;
            dispersed_shape_prep.reserve(block_length + 1);
            if (b_idx > 0)  // avoid addind empty Slice into Concat
                dispersed_shape_prep.push_back(rg.make<ov::op::v8::Slice>(squeezed_shape, zero, block_index, one));
            const auto squeezed_element = rg.make<ov::op::v8::Gather>(squeezed_shape, block_index, zero);
            dispersed_shape_prep.push_back(rg.make<ov::op::v1::Divide>(squeezed_element, block_value));
            dispersed_shape_prep.push_back(block_value);
            if (b_idx + 1 < block_length)  // avoid addind empty Slice into Concat
                dispersed_shape_prep.push_back(
                    rg.make<ov::op::v8::Slice>(squeezed_shape, block_index_next, int_max, one));

            const auto dispersed_shape = rg.make<ov::op::v0::Concat>(dispersed_shape_prep, 0);
            constexpr auto special_zero = false;
            flat_node = rg.make<ov::op::v1::Reshape>(flat_node, dispersed_shape, special_zero);

            vector<int64_t> axes_order(block_length + 1);
            int64_t axis_idx = axes_order.size() - 1;
            for (int64_t ds_idx = block_length; ds_idx >= 0; --ds_idx) {
                if (ds_idx == (b_idx + 1)) {
                    axes_order[0] = ds_idx;
                } else if (ds_idx == b_idx) {
                    axes_order[axis_idx] = ds_idx;
                    axis_idx--;
                } else {
                    axes_order[axis_idx] = ds_idx;
                    axis_idx--;
                }
            }
            const auto axes_order_const = rg.make<ov::op::v0::Constant>(i64, Shape{axes_order.size()}, axes_order);
            flat_node = rg.make<ov::op::v1::Transpose>(flat_node, axes_order_const);

            // don't change squeezed_shape at the last iteration, block[0] is assumed to be 1 by op definion
            if (b_idx > 0) {
                NodeVector squeezed_shape_prep;
                squeezed_shape_prep.reserve(block_length);
                squeezed_shape_prep.push_back(
                    rg.make<ov::op::v1::Multiply>(rg.make<ov::op::v8::Gather>(squeezed_shape, zero, zero),
                                                  block_value));
                if (b_idx > 1) {  // avoid addind empty Slice into Concat
                    squeezed_shape_prep.push_back(rg.make<ov::op::v8::Slice>(squeezed_shape, one, block_index, one));
                }
                squeezed_shape_prep.push_back(
                    rg.make<ov::op::v1::Divide>(rg.make<ov::op::v8::Gather>(squeezed_shape, block_index, zero),
                                                block_value));
                if (b_idx + 1 < block_length) {  // avoid addind empty Slice into Concat
                    squeezed_shape_prep.push_back(
                        rg.make<ov::op::v8::Slice>(squeezed_shape, block_index_next, int_max, one));
                }

                squeezed_shape = rg.make<ov::op::v0::Concat>(squeezed_shape_prep, 0);
            }
            flat_node = rg.make<ov::op::v1::Reshape>(flat_node, squeezed_shape, special_zero);
        }

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        copy_runtime_info(space_to_batch, rg.get());
        replace_node(space_to_batch, flat_node);
        return true;
    };

    const auto m = make_shared<pattern::Matcher>(space_to_batch, matcher_name);
    this->register_matcher(m, callback);
}
