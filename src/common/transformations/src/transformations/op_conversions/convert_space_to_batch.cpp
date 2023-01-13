// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_space_to_batch.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "itt.hpp"

void ov::pass::ConvertSpaceToBatch::convert_space_to_batch() {
    MATCHER_SCOPE(ConvertSpaceToBatch_convert_space_to_batch);
    auto space_to_batch = ngraph::pattern::wrap_type<ov::opset3::SpaceToBatch>();
    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto space_to_batch = std::dynamic_pointer_cast<ov::opset3::SpaceToBatch>(m.get_match_root());
        if (!space_to_batch) {
            return false;
        }

        NodeVector new_ops;
        auto data = space_to_batch->input_value(0);
        auto block = space_to_batch->input_value(1);
        auto pads_begin = space_to_batch->input_value(2);
        auto pads_end = space_to_batch->input_value(3);

        if (data.get_partial_shape().is_dynamic()) {
            return false;
        }

        const auto block_const = std::dynamic_pointer_cast<opset3::Constant>(block.get_node_shared_ptr());
        const auto pads_begin_const = std::dynamic_pointer_cast<opset3::Constant>(pads_begin.get_node_shared_ptr());
        const auto pads_end_const = std::dynamic_pointer_cast<opset3::Constant>(pads_end.get_node_shared_ptr());

        if (!block_const || !pads_begin_const || !pads_end_const) {
            return false;
        }

        const std::vector<int64_t>& block_values = block_const->cast_vector<int64_t>();

        //    Zero-pad the start and end of dimensions [D_0, ..., D_{N - 1}] of the input according to
        //    `pads_begin`
        //    and `pads_end`:
        //    note: P_0 for batch dimension is expected to be 0 (no-padding).
        //      x = [batch + P_0, D_1 + P_1, D_2 + P_2, ..., D_{N - 1} + P_{N - 1}], where P_i =
        //      pads_begin[i] + pads_end[i]
        std::shared_ptr<Node> flat_node =
            std::make_shared<opset3::Pad>(data, pads_begin_const, pads_end_const, ngraph::op::PadMode::CONSTANT);
        auto out_shape = flat_node->get_shape();
        new_ops.push_back(flat_node);

        // First we have to disperse the data from spatial dimensions, then
        // rearrange them so as appropriate chunks of data where close to their
        // destination place. Finally squeeze data from respective dimensions.
        Shape dispersed_shape{out_shape.at(0)};

        //    note: B_0 for batch is ignored.
        //      x' = reshape(x, [batch, (D_1 + P_1) / B_1, B_1, (D_2 + P_2) / B_2, B_2, ...,
        //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}, B_{N - 1}]), where B_i = block_shape[i]
        for (size_t i = 1; i < block_values.size(); ++i) {
            dispersed_shape.push_back(out_shape.at(i) / block_values.at(i));
            dispersed_shape.push_back(block_values.at(i));
        }

        const auto out_pattern = opset3::Constant::create(element::i64, Shape{dispersed_shape.size()}, dispersed_shape);
        flat_node = std::make_shared<ov::opset3::Reshape>(flat_node, out_pattern, false);
        new_ops.push_back(flat_node);

        //    x'' = transpose(x',  [2, 4, ..., (N - 1) + (N - 1), 0, 1, 3, ..., N + (N - 1)])
        std::vector<size_t> axes_order;
        for (size_t i = 0, j = 2; i < block_values.size() - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }
        axes_order.push_back(0);
        for (size_t i = 0, j = 1; i < block_values.size() - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }

        const auto axes_order_const =
            opset3::Constant::create(element::i64,
                                     Shape{axes_order.size()},
                                     std::vector<int64_t>(axes_order.begin(), axes_order.end()));
        flat_node = std::make_shared<ov::opset3::Transpose>(flat_node, axes_order_const);
        new_ops.push_back(flat_node);

        Shape squeezed_shape;
        int64_t prod = 1;
        for (const auto& el : block_values) {
            prod *= el;
        }

        //    y = reshape(x'', [batch * B_1 * ... * B_{N - 1}, (D_1 + P_1) / B_1, (D_2 + P_2) / B_2, ...
        //    ,
        //      (D_{N - 1} + P_{N - 1}) / B_{N - 1}])
        squeezed_shape.push_back(out_shape.at(0) * prod);
        for (size_t i = 1; i < block_values.size(); ++i) {
            squeezed_shape.push_back(out_shape.at(i) / block_values.at(i));
        }

        const auto out_pattern_2 = opset3::Constant::create(element::i64, Shape{squeezed_shape.size()}, squeezed_shape);
        flat_node = std::make_shared<ov::opset3::Reshape>(flat_node, out_pattern_2, false);
        new_ops.push_back(flat_node);

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        ngraph::copy_runtime_info(space_to_batch, new_ops);
        ngraph::replace_node(space_to_batch, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(space_to_batch, matcher_name);
    this->register_matcher(m, callback);
}

void ov::pass::ConvertSpaceToBatch::convert_space_to_batch_by_elements() {
    MATCHER_SCOPE(ConvertSpaceToBatch_convert_space_to_batch_by_elements);
    auto space_to_batch = ngraph::pattern::wrap_type<ov::opset3::SpaceToBatch>();
    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto space_to_batch = std::dynamic_pointer_cast<ov::opset3::SpaceToBatch>(m.get_match_root());
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

        const auto block_const = ov::as_type_ptr<opset3::Constant>(block.get_node_shared_ptr());

        if (!block_const) {
            return false;
        }
        const std::vector<int64_t>& block_values = block_const->cast_vector<int64_t>();

        NodeVector new_ops;

        std::shared_ptr<Node> flat_node =
            std::make_shared<opset3::Pad>(data, pads_begin, pads_end, ngraph::op::PadMode::CONSTANT);
        new_ops.push_back(flat_node);

        std::shared_ptr<Node> squeezed_shape = std::make_shared<opset3::ShapeOf>(flat_node);
        const auto block_lenght = block_values.size();
        OutputVector dispersed_shape(block_lenght + 1);
        std::vector<size_t> axes_order(block_lenght + 1);

        const auto zero = opset3::Constant::create(element::i64, Shape{1}, {0});
        const auto one = opset3::Constant::create(element::i64, Shape{1}, {1});
        for (int64_t block_idx = block_lenght - 1; block_idx >= 0; --block_idx) {
            int64_t sq_shape_idx = block_lenght - 1;
            int64_t axis_idx = axes_order.size() - 1;
            for (int64_t shape_idx = dispersed_shape.size() - 1; shape_idx >= 0; --shape_idx) {
                if (shape_idx == (block_idx + 1)) {
                    dispersed_shape[shape_idx] =
                        opset3::Constant::create(element::i64, Shape{1}, {block_values[block_idx]});
                    axes_order[0] = shape_idx;
                } else if (shape_idx == block_idx) {
                    const auto squeezed_element = std::make_shared<opset8::Gather>(
                        squeezed_shape,
                        opset3::Constant::create(element::i64, Shape{1}, {sq_shape_idx}),
                        zero);
                    dispersed_shape[shape_idx] = std::make_shared<opset3::Divide>(
                        squeezed_element,
                        opset3::Constant::create(element::i64, Shape{1}, {block_values[block_idx]}));
                    axes_order[axis_idx] = shape_idx;
                    axis_idx--;
                    sq_shape_idx--;
                } else {
                    dispersed_shape[shape_idx] = std::make_shared<opset8::Gather>(
                        squeezed_shape,
                        opset3::Constant::create(element::i64, Shape{1}, {sq_shape_idx}),
                        zero);
                    axes_order[axis_idx] = shape_idx;
                    axis_idx--;
                    sq_shape_idx--;
                }
            }

            const auto out_pattern_1 = std::make_shared<opset3::Concat>(dispersed_shape, 0);
            const bool special_zero = false;
            flat_node = std::make_shared<ov::opset3::Reshape>(flat_node, out_pattern_1, special_zero);
            new_ops.push_back(flat_node);

            const auto axes_order_const =
                opset3::Constant::create(element::i64,
                                         Shape{axes_order.size()},
                                         std::vector<int64_t>(axes_order.begin(), axes_order.end()));
            flat_node = std::make_shared<ov::opset3::Transpose>(flat_node, axes_order_const);
            new_ops.push_back(flat_node);

            const auto block_value = opset3::Constant::create(element::i64, Shape{1}, {block_values[block_idx]});
            // don't change squeezed_shape at the last iteration, block[0] is assumed to be 1 by op definion
            if (block_idx > 0) {
                OutputVector squeezed_shape_prep;
                squeezed_shape_prep.push_back(
                    std::make_shared<ov::opset3::Multiply>(std::make_shared<opset8::Gather>(squeezed_shape, zero, zero),
                                                           block_value));
                if (block_idx > 1) {
                    squeezed_shape_prep.push_back(
                        std::make_shared<opset8::Slice>(squeezed_shape,
                                                        one,
                                                        opset3::Constant::create(element::i64, Shape{1}, {block_idx}),
                                                        one));
                }
                squeezed_shape_prep.push_back(std::make_shared<opset3::Divide>(
                    std::make_shared<opset8::Gather>(squeezed_shape,
                                                     opset3::Constant::create(element::i64, Shape{1}, {block_idx}),
                                                     zero),
                    block_value));
                if (block_idx < block_lenght - 1) {
                    squeezed_shape_prep.push_back(std::make_shared<opset8::Slice>(
                        squeezed_shape,
                        opset3::Constant::create(element::i64, Shape{1}, {block_idx + 1}),
                        opset3::Constant::create(element::i64, Shape{1}, {block_lenght}),
                        one));
                }

                squeezed_shape = std::make_shared<opset3::Concat>(squeezed_shape_prep, 0);
            }
            flat_node = std::make_shared<ov::opset3::Reshape>(flat_node, squeezed_shape, special_zero);
            new_ops.push_back(flat_node);
        }

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        ngraph::copy_runtime_info(space_to_batch, new_ops);
        ngraph::replace_node(space_to_batch, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(space_to_batch, matcher_name);
    this->register_matcher(m, callback);
}
