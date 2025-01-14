// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_batch_to_space.hpp"

#include <algorithm>
#include <climits>
#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace std;
using namespace ov::element;

void ov::pass::ConvertBatchToSpace::convert_batch_to_space() {
    MATCHER_SCOPE(ConvertBatchToSpace_convert_batch_to_space);
    const auto batch_to_space = pattern::wrap_type<ov::op::v1::BatchToSpace>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto batch_to_space = ov::as_type_ptr<ov::op::v1::BatchToSpace>(m.get_match_root());
        if (!batch_to_space || transformation_callback(batch_to_space)) {
            return false;
        }

        NodeRegistry rg;
        const auto data = batch_to_space->input_value(0);
        const auto block = batch_to_space->input_value(1);
        const auto crops_begin = batch_to_space->input_value(2);
        const auto crops_end = batch_to_space->input_value(3);

        const auto data_shape_rank = data.get_partial_shape().rank();
        if (data_shape_rank.is_dynamic()) {
            return false;  // because StridedSlice masks are std::vector
        }

        if (block.get_partial_shape().is_dynamic() || block.get_shape().size() == 0) {
            return false;
        }
        const auto block_length = static_cast<int64_t>(block.get_shape()[0]);

        // First we have to disperse the data from batch, then rearrange them
        // so as appropriate chunks of data where close to their destination place.
        // Finally squeeze data from respective dimensions

        const auto zero = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 0);
        const auto shape_of_data = rg.make<ov::op::v3::ShapeOf>(data, block.get_element_type());
        const auto batch = rg.make<ov::op::v8::Gather>(shape_of_data, zero, zero);
        const auto block_prod = rg.make<ov::op::v1::ReduceProd>(block, zero);
        const auto batch_div = rg.make<ov::op::v1::Divide>(batch, block_prod);

        //   note: B_0 is expected to be 1.
        //      x' = reshape(`data`, [B_1, ..., B_{N - 1}, batch / (B_1 * ... B_{N - 1}), D_1, D_2, ...,
        //      D_{N - 1}]),
        //      where B_i = block_shape[i]
        const auto one = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 1);
        const auto end = rg.make<ov::op::v0::Constant>(i64, Shape{1}, block_length);
        const auto block_tail = rg.make<ov::op::v8::Slice>(block, one, end, one);
        const auto data_shape_tail = rg.make<ov::op::v8::Slice>(shape_of_data, one, end, one);
        const auto dispersed_shape =
            rg.make<ov::op::v0::Concat>(OutputVector{block_tail, batch_div, data_shape_tail}, 0);
        const bool special_zero = false;
        shared_ptr<Node> flat_node = rg.make<ov::op::v1::Reshape>(data, dispersed_shape, special_zero);

        // calculate axes to transpose
        //      x'' = transpose(x', [N, N + 1, 0, N + 2, 1, ..., N + N - 1, N - 1])
        vector<int64_t> axes_order{block_length - 1};
        for (int64_t i = 0; i < block_length - 1; ++i) {
            axes_order.push_back(i + block_length);
            axes_order.push_back(i);
        }
        const auto axes_order_const = rg.make<ov::op::v0::Constant>(i64, Shape{axes_order.size()}, axes_order);
        flat_node = rg.make<ov::op::v1::Transpose>(flat_node, axes_order_const);

        //   x''' = reshape(x'', [batch / (B_1 * ... * B_{N - 1}), D_1 * B_1, D_2 * B_2, ... , D_{N - 1}
        //   * B_{N - 1}])
        const auto squeezed_shape_tail = rg.make<ov::op::v1::Multiply>(block_tail, data_shape_tail);
        const auto squeezed_shape = rg.make<ov::op::v0::Concat>(OutputVector{batch_div, squeezed_shape_tail}, 0);
        flat_node = rg.make<ov::op::v1::Reshape>(flat_node, squeezed_shape, special_zero);

        //    Crop the start and end of dimensions according to `crops_begin`, `crops_end` to produce
        //    the output of shape:
        //    note: `crops_begin[0], crops_end[0]` are expected to be 0.
        //    `y = [batch / (B_1 * ... * B_{N - 1}), crop(D_1 * B_1, crops_begin[1], crops_end[1]),
        //          crop(D_2 * B_2, crops_begin[2], crops_end[2]), ... ,
        //          crop(D_{N - 1} * B_{N - 1}, crops_begin[N - 1], crops_end[N - 1])]`
        const auto shape_of_flat_node = rg.make<ov::op::v3::ShapeOf>(flat_node, crops_end.get_element_type());
        const auto upperbounds = rg.make<ov::op::v1::Subtract>(shape_of_flat_node, crops_end);

        const auto begin_mask = vector<int64_t>(data_shape_rank.get_length(), 0);
        const auto& end_mask = begin_mask;
        flat_node = rg.make<ov::op::v1::StridedSlice>(flat_node, crops_begin, upperbounds, begin_mask, end_mask);

        flat_node->set_friendly_name(batch_to_space->get_friendly_name());
        copy_runtime_info(batch_to_space, rg.get());
        replace_node(batch_to_space, flat_node);
        return true;
    };

    const auto m = make_shared<pattern::Matcher>(batch_to_space, matcher_name);
    this->register_matcher(m, callback);
}

void ov::pass::ConvertBatchToSpace::convert_batch_to_space_by_elements() {
    MATCHER_SCOPE(ConvertBatchToSpace_convert_batch_to_space_by_elements);
    const auto batch_to_space = pattern::wrap_type<ov::op::v1::BatchToSpace>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto batch_to_space = ov::as_type_ptr<ov::op::v1::BatchToSpace>(m.get_match_root());
        if (!batch_to_space || transformation_callback(batch_to_space)) {
            return false;
        }

        const auto data = batch_to_space->input_value(0);

        const auto data_shape_rank = data.get_partial_shape().rank();
        if (data_shape_rank.is_dynamic()) {
            return false;  // because StridedSlice masks are std::vector
        }

        const auto block = batch_to_space->input_value(1);
        const auto crops_begin = batch_to_space->input_value(2);
        const auto crops_end = batch_to_space->input_value(3);

        if (block.get_partial_shape().is_dynamic() || block.get_shape().size() == 0) {
            return false;
        }
        const auto block_length = static_cast<int64_t>(block.get_shape()[0]);

        NodeRegistry rg;
        const auto zero = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 0);
        const auto one = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 1);
        const auto two = rg.make<ov::op::v0::Constant>(i64, Shape{1}, 2);
        const auto int_max = rg.make<ov::op::v0::Constant>(i64, Shape{1}, INT_MAX);

        const auto shape_of_data = rg.make<ov::op::v3::ShapeOf>(data, block.get_element_type());
        const auto et_zero = rg.make<ov::op::v0::Constant>(block.get_element_type(), Shape{1}, 0);
        shared_ptr<Node> dispersed_shape = rg.make<ov::op::v0::Concat>(OutputVector{et_zero, shape_of_data}, 0);
        shared_ptr<Node> squeezed_shape = shape_of_data;

        shared_ptr<Node> flat_node = data.get_node_shared_ptr();

        const auto make_concat = [&](OutputVector nodes) {
            nodes.erase(remove_if(nodes.begin(),
                                  nodes.end(),
                                  [](const Output<Node>& n) {
                                      return n.get_partial_shape().is_static() && n.get_shape().size() > 0 &&
                                             n.get_shape()[0] == 0;
                                  }),
                        nodes.end());
            return rg.make<ov::op::v0::Concat>(nodes, 0);
        };

        shared_ptr<Node> div;
        for (int64_t b_idx = 1; b_idx < block_length; ++b_idx) {
            const auto block_index = rg.make<ov::op::v0::Constant>(i64, Shape{1}, b_idx);
            const auto block_index_next = rg.make<ov::op::v0::Constant>(i64, Shape{1}, b_idx + 1);
            const auto block_value = rg.make<ov::op::v8::Gather>(block, block_index, zero);

            // dispersed_shape[0] = block[b_idx];
            // dispersed_shape[1] /= block[b_idx];
            if (!div) {
                const auto batch = rg.make<ov::op::v8::Gather>(shape_of_data, zero, zero);
                div = rg.make<ov::op::v1::Divide>(batch, block_value);
            } else {
                div = rg.make<ov::op::v1::Divide>(div, block_value);
            }
            auto ds_tail = rg.make<ov::op::v8::Slice>(dispersed_shape, two, int_max, one);
            dispersed_shape = make_concat({block_value, div, ds_tail});
            constexpr auto special_zero = false;
            flat_node = rg.make<ov::op::v1::Reshape>(flat_node, dispersed_shape, special_zero);

            vector<int64_t> axes_order(block_length + 1);
            int64_t val = 1;
            for (int64_t axis_idx = 0; axis_idx <= block_length; ++axis_idx) {
                if ((b_idx + 1) == axis_idx) {
                    axes_order[axis_idx] = 0;
                } else {
                    axes_order[axis_idx] = val;
                    val++;
                }
            }
            const auto axes_order_const = rg.make<ov::op::v0::Constant>(i64, Shape{axes_order.size()}, axes_order);
            flat_node = rg.make<ov::op::v1::Transpose>(flat_node, axes_order_const);

            // squeezed_shape[0] = dispersed_shape[1];
            // squeezed_shape[b_idx] *= block[b_idx];
            const auto sq_slice = rg.make<ov::op::v8::Slice>(squeezed_shape, one, block_index, one);
            const auto sq_bidx_dim = rg.make<ov::op::v8::Gather>(squeezed_shape, block_index, zero);
            const auto sq_mul = rg.make<ov::op::v1::Multiply>(sq_bidx_dim, block_value);
            const auto sq_shape_tail = rg.make<ov::op::v8::Slice>(squeezed_shape, block_index_next, int_max, one);
            squeezed_shape.reset();
            squeezed_shape = make_concat({div, sq_slice, sq_mul, sq_shape_tail});
            flat_node = rg.make<ov::op::v1::Reshape>(flat_node, squeezed_shape, special_zero);

            // dispersed_shape[b_idx + 1] = squeezed_shape[b_idx];
            const auto ds_front = rg.make<ov::op::v8::Slice>(dispersed_shape, zero, block_index_next, one);
            ds_tail = rg.make<ov::op::v8::Slice>(dispersed_shape,
                                                 rg.make<ov::op::v0::Constant>(i64, Shape{1}, b_idx + 2),
                                                 int_max,
                                                 one);
            dispersed_shape = make_concat({ds_front, sq_mul, ds_tail});
        }

        const auto shape_of_flat_node = rg.make<ov::op::v3::ShapeOf>(flat_node, crops_end.get_element_type());
        const auto upperbounds = rg.make<ov::op::v1::Subtract>(shape_of_flat_node, crops_end);

        const auto begin_mask = vector<int64_t>(data_shape_rank.get_length(), 0);
        const auto& end_mask = begin_mask;
        flat_node = rg.make<ov::op::v1::StridedSlice>(flat_node, crops_begin, upperbounds, begin_mask, end_mask);

        flat_node->set_friendly_name(batch_to_space->get_friendly_name());
        copy_runtime_info(batch_to_space, rg.get());
        replace_node(batch_to_space, flat_node);
        return true;
    };

    const auto m = make_shared<pattern::Matcher>(batch_to_space, matcher_name);
    this->register_matcher(m, callback);
}
