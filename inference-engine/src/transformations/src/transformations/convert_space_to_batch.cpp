// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_space_to_batch.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/builder/reshape.hpp>

void ngraph::pass::ConvertSpaceToBatch::convert_space_to_batch() {
    auto input0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input1 = ngraph::opset2::Constant::create(element::i64, Shape{4}, Shape{1, 1, 1, 1});
    auto input2 = ngraph::opset2::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto input3 = ngraph::opset2::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto space_to_batch = std::make_shared<ngraph::opset2::SpaceToBatch>(input0, input1, input2, input3);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto space_to_batch = std::dynamic_pointer_cast<ngraph::opset2::SpaceToBatch> (m.get_match_root());
        if (!space_to_batch) {
            return false;
        }

        auto data = space_to_batch->input_value(0);
        auto block = space_to_batch->input_value(1);
        auto pads_begin = space_to_batch->input_value(2);
        auto pads_end = space_to_batch->input_value(3);

        const auto& data_shape = data.get_shape();

        const auto block_const = std::dynamic_pointer_cast<opset2::Constant>(block.get_node_shared_ptr());
        const auto pads_begin_const = std::dynamic_pointer_cast<opset2::Constant>(pads_begin.get_node_shared_ptr());
        const auto pads_end_const = std::dynamic_pointer_cast<opset2::Constant>(pads_end.get_node_shared_ptr());

        if (!block_const || !pads_begin_const || !pads_end_const) {
            return false;
        }

        std::vector<int64_t> block_values;
        block_values = block_const->cast_vector<int64_t>();

        //    Zero-pad the start and end of dimensions [D_0, ..., D_{N - 1}] of the input according to
        //    `pads_begin`
        //    and `pads_end`:
        //    note: P_0 for batch dimension is expected to be 0 (no-padding).
        //      x = [batch + P_0, D_1 + P_1, D_2 + P_2, ..., D_{N - 1} + P_{N - 1}], where P_i =
        //      pads_begin[i] + pads_end[i]
        auto out = std::make_shared<opset2::Pad>(data, pads_begin_const, pads_end_const, ngraph::op::PadMode::CONSTANT);
        auto out_shape = out->get_shape();

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
        auto flat_node = builder::opset1::reshape(out, dispersed_shape);

        //    x'' = transpose(x',  [2, 4, ..., (N - 1) + (N - 1), 0, 1, 3, ..., N + (N - 1)])
        std::vector<size_t> axes_order;
        for (size_t i = 0, j = 2; i < block_values.size() - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }
        axes_order.push_back(0);
        for (size_t i = 0, j = 1; i < block_values.size() - 1; ++i, j += 2) {
            axes_order.push_back(j);
        }

        flat_node = builder::opset1::reorder_axes(flat_node, axes_order);
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
        flat_node = ngraph::builder::opset1::reshape(flat_node, squeezed_shape);

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        ngraph::copy_runtime_info(space_to_batch, flat_node);
        ngraph::replace_node(space_to_batch, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(space_to_batch, "ConvertSpaceToBatch");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

void ngraph::pass::ConvertSpaceToBatch::convert_space_to_batch_by_elements() {
    auto input0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input1 = ngraph::op::Constant::create(element::i64, Shape{4}, Shape{1, 1, 1, 1});
    auto input2 = ngraph::op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto input3 = ngraph::op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto space_to_batch = std::make_shared<ngraph::opset2::SpaceToBatch>(input0, input1, input2, input3);

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto space_to_batch = std::dynamic_pointer_cast<ngraph::opset2::SpaceToBatch> (m.get_match_root());
        if (!space_to_batch) {
            return false;
        }

        auto data = space_to_batch->input_value(0);
        const auto& data_shape = data.get_shape();

        if (transformation_callback(space_to_batch) && (data_shape.size() == 4 || data_shape.size() == 5)) {
            return false;
        }

        auto block = space_to_batch->input_value(1);
        auto pads_begin = space_to_batch->input_value(2);
        auto pads_end = space_to_batch->input_value(3);

        const auto block_const = as_type_ptr<op::Constant>(block.get_node_shared_ptr());
        const auto pads_begin_const = as_type_ptr<op::Constant>(pads_begin.get_node_shared_ptr());
        const auto pads_end_const = as_type_ptr<op::Constant>(pads_end.get_node_shared_ptr());

        std::vector<int64_t> block_values;
        block_values = block_const->cast_vector<int64_t>();

        NodeVector new_ops;

        std::shared_ptr<Node> flat_node = data.get_node_shared_ptr();
        flat_node = std::make_shared<opset2::Pad>(flat_node, pads_begin_const, pads_end_const, ngraph::op::PadMode::CONSTANT);
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
                    op::Constant::create(element::i64, Shape{dispersed_shape.size()}, dispersed_shape);
            const bool special_zero = false;
            flat_node = std::make_shared<ngraph::op::v1::Reshape>(flat_node, out_pattern_1, special_zero);
            new_ops.push_back(flat_node);

            const auto axes_order_const =
                    op::Constant::create(element::i64,
                                         Shape{axes_order.size()},
                                         std::vector<int64_t>(axes_order.begin(), axes_order.end()));
            flat_node = std::make_shared<ngraph::opset1::Transpose>(flat_node, axes_order_const)
                    ->add_provenance_group_members_above({flat_node});
            new_ops.push_back(flat_node);
            squeezed_shape[0] *= block_values[block_idx];
            squeezed_shape[block_idx] /= block_values[block_idx];
            const auto out_pattern_2 =
                    op::Constant::create(element::i64, Shape{squeezed_shape.size()}, squeezed_shape);
            flat_node = std::make_shared<ngraph::op::v1::Reshape>(flat_node, out_pattern_2, special_zero)
                    ->add_provenance_group_members_above({data});
            new_ops.push_back(flat_node);
        }

        flat_node->set_friendly_name(space_to_batch->get_friendly_name());
        ngraph::copy_runtime_info(space_to_batch, new_ops);
        ngraph::replace_node(space_to_batch, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(space_to_batch, "ConvertSpaceToBatch");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}