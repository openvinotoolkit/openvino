// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_space_to_batch.hpp"

#include <climits>
#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/opsets/opset3.hpp>
#include <vector>

#include "itt.hpp"

using namespace std;
using namespace ov::opset10;
using namespace ov::element;

void ov::pass::ConvertSpaceToBatch::convert_space_to_batch() {
    MATCHER_SCOPE(ConvertSpaceToBatch_convert_space_to_batch);
    const auto space_to_batch = pattern::wrap_type<opset3::SpaceToBatch>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto space_to_batch = dynamic_pointer_cast<opset3::SpaceToBatch>(m.get_match_root());
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

        // static data shape rank implies static block shape
        const auto block_lenght = static_cast<int64_t>(block.get_shape()[0]);

        NodeRegistry rg;

        //    Zero-pad the start and end of dimensions [D_0, ..., D_{N - 1}] of the input according to
        //    `pads_begin`
        //    and `pads_end`:
        //    note: P_0 for batch dimension is expected to be 0 (no-padding).
        //      x = [batch + P_0, D_1 + P_1, D_2 + P_2, ..., D_{N - 1} + P_{N - 1}], where P_i =
        //      pads_begin[i] + pads_end[i]
        shared_ptr<Node> flat_node = rg.make<Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);
        const auto out_shape = rg.make<ShapeOf>(flat_node);

        const auto zero = rg.make<Constant>(i64, Shape{1}, 0);

        // First we have to disperse the data from spatial dimensions, then
        // rearrange them so as appropriate chunks of data where close to their
        // destination place. Finally squeeze data from respective dimensions.

        //    note: B_0 for batch is ignored.
        //      x' = reshape(x, [batch, (D_1 + P_1) / B_1, B_1, (D_2 + P_2) / B_2, B_2, ...,
        //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}, B_{N - 1}]), where B_i = block_shape[i]
        const auto batch = rg.make<Gather>(out_shape, zero, zero);
        NodeVector dispersed_shape_prep{batch};

        //    y = reshape(x'', [batch * B_1 * ... * B_{N - 1}, (D_1 + P_1) / B_1, (D_2 + P_2) / B_2, ...,
        //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}])
        //    note: B_0 is assumed to be 1 by op definion
        const auto block_prod = rg.make<ReduceProd>(block, zero);
        NodeVector squeezed_shape_prep{rg.make<Multiply>(batch, block_prod)};

        for (int64_t idx = 1; idx < block_lenght; ++idx) {
            const auto index = rg.make<Constant>(i64, Shape{1}, idx);
            const auto out_dim = rg.make<Gather>(out_shape, index, zero);
            const auto block_value = rg.make<Gather>(block, index, zero);
            const auto div = rg.make<Divide>(out_dim, block_value);

            dispersed_shape_prep.push_back(div);
            dispersed_shape_prep.push_back(block_value);

            squeezed_shape_prep.push_back(div);
        }

        const auto dispersed_shape = rg.make<Concat>(dispersed_shape_prep, 0);
        flat_node = rg.make<Reshape>(flat_node, dispersed_shape, false);

        //    x'' = transpose(x',  [2, 4, ..., (N - 1) + (N - 1), 0, 1, 3, ..., N + (N - 1)])
        vector<int64_t> axes_order;
        for (int64_t i = 0, j = 2; i < block_lenght - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }
        axes_order.push_back(0);
        for (int64_t i = 0, j = 1; i < block_lenght - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }

        const auto axes_order_const = rg.make<Constant>(i64, Shape{axes_order.size()}, axes_order);
        flat_node = rg.make<Transpose>(flat_node, axes_order_const);

        const auto squeezed_shape = rg.make<Concat>(squeezed_shape_prep, 0);
        flat_node = rg.make<Reshape>(flat_node, squeezed_shape, false);

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
    const auto space_to_batch = pattern::wrap_type<opset3::SpaceToBatch>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto space_to_batch = dynamic_pointer_cast<opset3::SpaceToBatch>(m.get_match_root());
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

        // static data shape rank implies static block shape
        const auto block_lenght = static_cast<int64_t>(block.get_shape()[0]);

        NodeRegistry rg;

        shared_ptr<Node> flat_node = rg.make<Pad>(data, pads_begin, pads_end, op::PadMode::CONSTANT);

        shared_ptr<Node> squeezed_shape = rg.make<ShapeOf>(flat_node);
        vector<int64_t> axes_order(block_lenght + 1);

        const auto zero = rg.make<Constant>(i64, Shape{1}, 0);
        const auto one = rg.make<Constant>(i64, Shape{1}, 1);
        const auto int_max = rg.make<Constant>(i64, Shape{1}, INT_MAX);

        for (int64_t b_idx = block_lenght - 1; b_idx >= 0; --b_idx) {
            const auto block_index = rg.make<Constant>(i64, Shape{1}, b_idx);
            const auto block_index_next = rg.make<Constant>(i64, Shape{1}, b_idx + 1);
            const auto block_value = rg.make<Gather>(block, block_index, zero);
            int64_t sq_idx = block_lenght - 1;
            int64_t axis_idx = axes_order.size() - 1;

            NodeVector dispersed_shape_prep(block_lenght + 1);
            for (int64_t ds_idx = block_lenght; ds_idx >= 0; --ds_idx) {
                const auto squeezed_index = rg.make<Constant>(i64, Shape{1}, sq_idx);
                if (ds_idx == (b_idx + 1)) {
                    dispersed_shape_prep[ds_idx] = block_value;
                    axes_order[0] = ds_idx;
                } else if (ds_idx == b_idx) {
                    const auto squeezed_element = rg.make<Gather>(squeezed_shape, squeezed_index, zero);
                    dispersed_shape_prep[ds_idx] = rg.make<Divide>(squeezed_element, block_value);
                    axes_order[axis_idx] = ds_idx;
                    axis_idx--;
                    sq_idx--;
                } else {
                    dispersed_shape_prep[ds_idx] = rg.make<Gather>(squeezed_shape, squeezed_index, zero);
                    axes_order[axis_idx] = ds_idx;
                    axis_idx--;
                    sq_idx--;
                }
            }

            const auto dispersed_shape = rg.make<Concat>(dispersed_shape_prep, 0);
            constexpr auto special_zero = false;
            flat_node = rg.make<Reshape>(flat_node, dispersed_shape, special_zero);

            const auto axes_order_const = rg.make<Constant>(i64, Shape{axes_order.size()}, axes_order);
            flat_node = rg.make<Transpose>(flat_node, axes_order_const);

            // don't change squeezed_shape at the last iteration, block[0] is assumed to be 1 by op definion
            if (b_idx > 0) {
                NodeVector squeezed_shape_prep;
                squeezed_shape_prep.push_back(
                    rg.make<Multiply>(rg.make<Gather>(squeezed_shape, zero, zero), block_value));
                if (b_idx > 1) {  // avoid addind empty Slice into Concat
                    squeezed_shape_prep.push_back(rg.make<Slice>(squeezed_shape, one, block_index, one));
                }
                squeezed_shape_prep.push_back(
                    rg.make<Divide>(rg.make<Gather>(squeezed_shape, block_index, zero), block_value));
                if (b_idx < block_lenght - 1) {  // avoid addind empty Slice into Concat
                    squeezed_shape_prep.push_back(rg.make<Slice>(squeezed_shape, block_index_next, int_max, one));
                }

                squeezed_shape = rg.make<Concat>(squeezed_shape_prep, 0);
            }
            flat_node = rg.make<Reshape>(flat_node, squeezed_shape, special_zero);
        }

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        copy_runtime_info(space_to_batch, rg.get());
        replace_node(space_to_batch, flat_node);
        return true;
    };

    const auto m = make_shared<pattern::Matcher>(space_to_batch, matcher_name);
    this->register_matcher(m, callback);
}
