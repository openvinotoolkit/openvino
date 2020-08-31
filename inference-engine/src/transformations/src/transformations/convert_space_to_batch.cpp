// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_space_to_batch.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertSpaceToBatch::convert_space_to_batch() {
    auto space_to_batch = ngraph::pattern::wrap_type<ngraph::opset3::SpaceToBatch>();
    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto space_to_batch = std::dynamic_pointer_cast<ngraph::opset3::SpaceToBatch> (m.get_match_root());
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
        const auto& data_shape = data.get_shape();

        const auto block_const = std::dynamic_pointer_cast<opset3::Constant>(block.get_node_shared_ptr());
        const auto pads_begin_const = std::dynamic_pointer_cast<opset3::Constant>(pads_begin.get_node_shared_ptr());
        const auto pads_end_const = std::dynamic_pointer_cast<opset3::Constant>(pads_end.get_node_shared_ptr());

        if (!block_const || !pads_begin_const || !pads_end_const) {
            return false;
        }

        const std::vector<int64_t> &block_values = block_const->cast_vector<int64_t>();

        //    Zero-pad the start and end of dimensions [D_0, ..., D_{N - 1}] of the input according to
        //    `pads_begin`
        //    and `pads_end`:
        //    note: P_0 for batch dimension is expected to be 0 (no-padding).
        //      x = [batch + P_0, D_1 + P_1, D_2 + P_2, ..., D_{N - 1} + P_{N - 1}], where P_i =
        //      pads_begin[i] + pads_end[i]
        std::shared_ptr<Node> flat_node = std::make_shared<opset3::Pad>(data, pads_begin_const, pads_end_const,
                ngraph::op::PadMode::CONSTANT);
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

        const auto out_pattern =
                opset3::Constant::create(element::i64, Shape{dispersed_shape.size()}, dispersed_shape);
        flat_node = std::make_shared<ngraph::opset3::Reshape>(flat_node, out_pattern, false);
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
        flat_node = std::make_shared<ngraph::opset3::Transpose>(flat_node, axes_order_const);
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

        const auto out_pattern_2 =
                opset3::Constant::create(element::i64, Shape{squeezed_shape.size()}, squeezed_shape);
        flat_node = std::make_shared<ngraph::opset3::Reshape>(flat_node, out_pattern_2, false);
        new_ops.push_back(flat_node);

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        ngraph::copy_runtime_info(space_to_batch, new_ops);
        ngraph::replace_node(space_to_batch, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(space_to_batch, "ConvertSpaceToBatch");
    this->register_matcher(m, callback);
}

void ngraph::pass::ConvertSpaceToBatch::convert_space_to_batch_by_elements() {
    auto space_to_batch = ngraph::pattern::wrap_type<ngraph::opset3::SpaceToBatch>();
    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto space_to_batch = std::dynamic_pointer_cast<ngraph::opset3::SpaceToBatch> (m.get_match_root());
        if (!space_to_batch) {
            return false;
        }

        auto data = space_to_batch->input_value(0);

        if (data.get_partial_shape().is_dynamic()) {
            return false;
        }
        const auto& data_shape = data.get_shape();

        if (m_transformation_callback(space_to_batch) && (data_shape.size() == 4 || data_shape.size() == 5)) {
            return false;
        }

        auto block = space_to_batch->input_value(1);
        auto pads_begin = space_to_batch->input_value(2);
        auto pads_end = space_to_batch->input_value(3);

        const auto block_const = as_type_ptr<opset3::Constant>(block.get_node_shared_ptr());
        const auto pads_begin_const = as_type_ptr<opset3::Constant>(pads_begin.get_node_shared_ptr());
        const auto pads_end_const = as_type_ptr<opset3::Constant>(pads_end.get_node_shared_ptr());

        if (!block_const || !pads_begin_const || !pads_end_const) {
            return false;
        }
        const std::vector<int64_t> &block_values = block_const->cast_vector<int64_t>();

        NodeVector new_ops;

        std::shared_ptr<Node> flat_node = std::make_shared<opset3::Pad>(data, pads_begin_const, pads_end_const, ngraph::op::PadMode::CONSTANT);
        new_ops.push_back(flat_node);
        auto out_shape = flat_node->get_shape();

        std::vector<int64_t> dispersed_shape(block_values.size() + 1);
        std::vector<size_t> axes_order(block_values.size() + 1);
        std::vector<int64_t> squeezed_shape(out_shape.begin(), out_shape.end());
        for (int64_t block_idx = block_values.size() - 1; block_idx >= 0; --block_idx) {
            int64_t sq_shape_idx = block_values.size() - 1;
            int64_t axis_idx = axes_order.size() - 1;
            for (int64_t shape_idx = dispersed_shape.size() - 1; shape_idx >= 0; --shape_idx) {
                if (shape_idx == (block_idx + 1)) {
                    dispersed_shape[shape_idx] = block_values[block_idx];
                    axes_order[0] = shape_idx;
                } else if (shape_idx == block_idx) {
                    dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx]/block_values[block_idx];
                    axes_order[axis_idx] = shape_idx;
                    axis_idx--;
                    sq_shape_idx--;
                } else {
                    dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx];
                    axes_order[axis_idx] = shape_idx;
                    axis_idx--;
                    sq_shape_idx--;
                }
            }

            const auto out_pattern_1 =
                    opset3::Constant::create(element::i64, Shape{dispersed_shape.size()}, dispersed_shape);
            const bool special_zero = false;
            flat_node = std::make_shared<ngraph::opset3::Reshape>(flat_node, out_pattern_1, special_zero);
            new_ops.push_back(flat_node);

            const auto axes_order_const =
                    opset3::Constant::create(element::i64,
                                         Shape{axes_order.size()},
                                         std::vector<int64_t>(axes_order.begin(), axes_order.end()));
            flat_node = std::make_shared<ngraph::opset3::Transpose>(flat_node, axes_order_const);
            new_ops.push_back(flat_node);
            squeezed_shape[0] *= block_values[block_idx];
            squeezed_shape[block_idx] /= block_values[block_idx];
            const auto out_pattern_2 =
                    opset3::Constant::create(element::i64, Shape{squeezed_shape.size()}, squeezed_shape);
            flat_node = std::make_shared<ngraph::opset3::Reshape>(flat_node, out_pattern_2, special_zero);
            new_ops.push_back(flat_node);
        }

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        ngraph::copy_runtime_info(space_to_batch, new_ops);
        ngraph::replace_node(space_to_batch, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(space_to_batch, "ConvertSpaceToBatch");
    this->register_matcher(m, callback);
}