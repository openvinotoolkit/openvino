// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_batch_to_space.hpp"

#include <algorithm>
#include <climits>
#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "itt.hpp"

using namespace ov;
using namespace opset10;
using namespace std;

void ov::pass::ConvertBatchToSpace::convert_batch_to_space() {
    MATCHER_SCOPE(ConvertBatchToSpace_convert_batch_to_space);
    auto batch_to_space = ngraph::pattern::wrap_type<ov::opset3::BatchToSpace>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto batch_to_space = std::dynamic_pointer_cast<ov::opset3::BatchToSpace>(m.get_match_root());
        if (!batch_to_space || transformation_callback(batch_to_space)) {
            return false;
        }

        NodeVector new_ops;
        const auto data = batch_to_space->input_value(0);
        const auto block = batch_to_space->input_value(1);
        const auto crops_begin = batch_to_space->input_value(2);
        const auto crops_end = batch_to_space->input_value(3);

        const auto data_shape_rank = data.get_partial_shape().rank();
        if (data_shape_rank.is_dynamic()) {
            return false;  // beacuse StridedSlice masks are std::vector
        }

        // static data shape rank implies static block shape
        const auto block_lenght = static_cast<int64_t>(block.get_shape()[0]);

        // First we have to disperse the data from batch, then rearrange them
        // so as appropriate chunks of data where close to their destination place.
        // Finally squeeze data from respective dimensions

        const auto zero = opset3::Constant::create(element::i64, Shape{1}, {0});
        const auto shape_of_data = std::make_shared<opset3::ShapeOf>(data);
        const auto batch = std::make_shared<opset8::Gather>(shape_of_data, zero, zero);
        const auto block_prod = std::make_shared<opset3::ReduceProd>(block, zero);
        const auto batch_div = std::make_shared<opset3::Divide>(batch, block_prod);
        new_ops.push_back(shape_of_data);
        new_ops.push_back(batch);
        new_ops.push_back(block_prod);
        new_ops.push_back(batch_div);

        //   note: B_0 is expected to be 1.
        //      x' = reshape(`data`, [B_1, ..., B_{N - 1}, batch / (B_1 * ... B_{N - 1}), D_1, D_2, ...,
        //      D_{N - 1}]),
        //      where B_i = block_shape[i]
        const auto one = opset3::Constant::create(element::i64, Shape{1}, {1});
        const auto end = opset3::Constant::create(element::i64, Shape{1}, {block_lenght});
        const auto block_tail = std::make_shared<opset8::Slice>(block, one, end, one);
        const auto data_shape_tail = std::make_shared<opset8::Slice>(shape_of_data, one, end, one);
        const auto dispersed_shape =
            std::make_shared<opset3::Concat>(OutputVector{block_tail, batch_div, data_shape_tail}, 0);
        const bool special_zero = false;
        std::shared_ptr<Node> flat_node = std::make_shared<ov::opset3::Reshape>(data, dispersed_shape, special_zero);
        new_ops.push_back(block_tail);
        new_ops.push_back(data_shape_tail);
        new_ops.push_back(dispersed_shape);
        new_ops.push_back(flat_node);

        // calculate axes to transpose
        //      x'' = transpose(x', [N, N + 1, 0, N + 2, 1, ..., N + N - 1, N - 1])
        std::vector<int64_t> axes_order{block_lenght - 1};
        for (int64_t i = 0; i < block_lenght - 1; ++i) {
            axes_order.push_back(i + block_lenght);
            axes_order.push_back(i);
        }
        const auto axes_order_const =
            opset3::Constant::create(element::i64,
                                     Shape{axes_order.size()},
                                     std::vector<int64_t>(axes_order.begin(), axes_order.end()));
        flat_node = std::make_shared<ov::opset3::Transpose>(flat_node, axes_order_const);
        new_ops.push_back(flat_node);

        //   x''' = reshape(x'', [batch / (B_1 * ... * B_{N - 1}), D_1 * B_1, D_2 * B_2, ... , D_{N - 1}
        //   * B_{N - 1}])
        const auto squeezed_shape_tail = std::make_shared<opset3::Multiply>(block_tail, data_shape_tail);
        const auto squeezed_shape = std::make_shared<opset3::Concat>(OutputVector{batch_div, squeezed_shape_tail}, 0);
        flat_node = std::make_shared<opset3::Reshape>(flat_node, squeezed_shape, special_zero);
        new_ops.push_back(squeezed_shape_tail);
        new_ops.push_back(squeezed_shape);
        new_ops.push_back(flat_node);

        //    Crop the start and end of dimensions according to `crops_begin`, `crops_end` to produce
        //    the output of shape:
        //    note: `crops_begin[0], crops_end[0]` are expected to be 0.
        //    `y = [batch / (B_1 * ... * B_{N - 1}), crop(D_1 * B_1, crops_begin[1], crops_end[1]),
        //          crop(D_2 * B_2, crops_begin[2], crops_end[2]), ... ,
        //          crop(D_{N - 1} * B_{N - 1}, crops_begin[N - 1], crops_end[N - 1])]`
        const auto shape_of_flat_node = std::make_shared<opset3::ShapeOf>(flat_node, crops_end.get_element_type());
        const auto upperbounds = std::make_shared<opset3::Subtract>(shape_of_flat_node, crops_end);
        new_ops.push_back(shape_of_flat_node);
        new_ops.push_back(upperbounds);

        const auto begin_mask = std::vector<int64_t>(data_shape_rank.get_length(), 0);
        const auto& end_mask = begin_mask;
        flat_node = std::make_shared<opset3::StridedSlice>(flat_node, crops_begin, upperbounds, begin_mask, end_mask);
        new_ops.push_back(flat_node);

        flat_node->set_friendly_name(batch_to_space->get_friendly_name());
        ngraph::copy_runtime_info(batch_to_space, new_ops);
        ngraph::replace_node(batch_to_space, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(batch_to_space, matcher_name);
    this->register_matcher(m, callback);
}

void ov::pass::ConvertBatchToSpace::convert_batch_to_space_by_elements() {
    MATCHER_SCOPE(ConvertBatchToSpace_convert_batch_to_space_by_elements);
    auto batch_to_space = ngraph::pattern::wrap_type<ov::opset3::BatchToSpace>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto batch_to_space = std::dynamic_pointer_cast<ov::opset3::BatchToSpace>(m.get_match_root());
        if (!batch_to_space || transformation_callback(batch_to_space)) {
            return false;
        }

        const auto data = batch_to_space->input_value(0);

        const auto data_shape_rank = data.get_partial_shape().rank();
        if (data_shape_rank.is_dynamic()) {
            return false;  // beacuse StridedSlice masks are std::vector
        }

        const auto block = batch_to_space->input_value(1);
        const auto crops_begin = batch_to_space->input_value(2);
        const auto crops_end = batch_to_space->input_value(3);

        // static data shape rank implies static block shape
        const auto block_lenght = static_cast<int64_t>(block.get_shape()[0]);

        const auto zero = Constant::create(element::i64, Shape{1}, {0});
        const auto one = Constant::create(element::i64, Shape{1}, {1});
        const auto two = Constant::create(element::i64, Shape{1}, {2});
        const auto int_max = Constant::create(element::i64, Shape{1}, {INT_MAX});

        const auto shape_of_data = make_shared<ShapeOf>(data);
        shared_ptr<Node> dispersed_shape = make_shared<Concat>(OutputVector{zero, shape_of_data}, 0);
        shared_ptr<Node> squeezed_shape = shape_of_data;

        NodeVector new_ops;
        shared_ptr<Node> flat_node = data.get_node_shared_ptr();

        auto make_shape_node = [](OutputVector nodes) {
            nodes.erase(remove_if(nodes.begin(),
                                  nodes.end(),
                                  [](const Output<Node>& n) {
                                      return n.get_shape()[0] == 0;
                                  }),
                        nodes.end());
            return make_shared<Concat>(nodes, 0);
        };

        shared_ptr<Node> div;
        for (size_t b_idx = 1; b_idx < block_lenght; ++b_idx) {
            const auto block_index = Constant::create(element::i64, Shape{1}, {b_idx});
            const auto block_index_next = Constant::create(element::i64, Shape{1}, {b_idx + 1});
            const auto block_value = make_shared<Gather>(block, block_index, zero);

            // dispersed_shape[0] = block[b_idx];
            // dispersed_shape[1] /= block[b_idx];
            if (!div) {
                const auto batch = make_shared<Gather>(shape_of_data, zero, zero);
                div = make_shared<Divide>(batch, block_value);
            } else {
                div = make_shared<Divide>(div, block_value);
            }
            auto ds_tail = make_shared<Slice>(dispersed_shape, two, int_max, one);
            dispersed_shape = make_shape_node({block_value, div, ds_tail});
            constexpr auto special_zero = false;
            flat_node = make_shared<Reshape>(flat_node, dispersed_shape, special_zero);
            new_ops.push_back(flat_node);

            vector<size_t> axes_order(block_lenght + 1);
            size_t val = 1;
            for (size_t axis_idx = 0; axis_idx <= block_lenght; ++axis_idx) {
                if ((b_idx + 1) == axis_idx) {
                    axes_order[axis_idx] = 0;
                } else {
                    axes_order[axis_idx] = val;
                    val++;
                }
            }
            const auto axes_order_const = Constant::create(element::i64,
                                                           Shape{axes_order.size()},
                                                           vector<int64_t>(axes_order.begin(), axes_order.end()));
            flat_node = make_shared<Transpose>(flat_node, axes_order_const);
            new_ops.push_back(flat_node);

            // squeezed_shape[0] = dispersed_shape[1];
            // squeezed_shape[b_idx] *= block[b_idx];
            const auto sq_slice = make_shared<Slice>(squeezed_shape, one, block_index, one);
            const auto sq_bidx_dim = make_shared<Gather>(squeezed_shape, block_index, zero);
            const auto sq_mul = make_shared<Multiply>(sq_bidx_dim, block_value);
            const auto sq_shape_tail = make_shared<Slice>(squeezed_shape, block_index_next, int_max, one);
            squeezed_shape.reset();
            squeezed_shape = make_shape_node({div, sq_slice, sq_mul, sq_shape_tail});
            flat_node = make_shared<Reshape>(flat_node, squeezed_shape, special_zero);

            // dispersed_shape[b_idx + 1] = squeezed_shape[b_idx];
            const auto ds_front = make_shared<Slice>(dispersed_shape, zero, block_index_next, one);
            ds_tail = make_shared<Slice>(dispersed_shape,
                                         Constant::create(element::i64, Shape{1}, {b_idx + 2}),
                                         int_max,
                                         one);
            dispersed_shape = make_shape_node({ds_front, sq_mul, ds_tail});
        }

        const auto shape_of_flat_node = make_shared<ShapeOf>(flat_node, crops_end.get_element_type());
        const auto upperbounds = make_shared<Subtract>(shape_of_flat_node, crops_end);

        const auto begin_mask = vector<int64_t>(data_shape_rank.get_length(), 0);
        const auto& end_mask = begin_mask;
        flat_node = make_shared<StridedSlice>(flat_node, crops_begin, upperbounds, begin_mask, end_mask);
        new_ops.push_back(flat_node);

        flat_node->set_friendly_name(batch_to_space->get_friendly_name());
        copy_runtime_info(batch_to_space, new_ops);
        replace_node(batch_to_space, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(batch_to_space, matcher_name);
    this->register_matcher(m, callback);
}
