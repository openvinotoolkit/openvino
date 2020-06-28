// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_batch_to_space.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset2.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/builder/reshape.hpp>

void ngraph::pass::ConvertBatchToSpace::convert_batch_to_space() {
    auto input0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input1 = ngraph::opset2::Constant::create(element::i64, Shape{4}, Shape{1, 1, 1, 1});
    auto input2 = ngraph::opset2::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto input3 = ngraph::opset2::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto batch_to_space = std::make_shared<ngraph::opset2::BatchToSpace>(input0, input1, input2, input3);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto batch_to_space = std::dynamic_pointer_cast<ngraph::opset2::BatchToSpace> (m.get_match_root());
        if (!batch_to_space) {
            return false;
        }

        auto data = batch_to_space->input_value(0);
        auto block = batch_to_space->input_value(1);
        auto crops_begin = batch_to_space->input_value(2);
        auto crops_end = batch_to_space->input_value(3);

        const auto& data_shape = data.get_shape();

        const auto block_const = std::dynamic_pointer_cast<opset2::Constant>(block.get_node_shared_ptr());
        const auto crops_begin_const = std::dynamic_pointer_cast<opset2::Constant>(crops_begin.get_node_shared_ptr());
        const auto crops_end_const = std::dynamic_pointer_cast<opset2::Constant>(crops_end.get_node_shared_ptr());

        if (!block_const || !crops_begin_const || !crops_end_const) {
            return false;
        }

        std::vector<int64_t> block_values, crops_end_values;
        block_values = block_const->cast_vector<int64_t>();
        crops_end_values = crops_end_const->cast_vector<int64_t>();

        // First we have to disperse the data from batch, then rearrange them
        // so as appropriate chunks of data where close to their destination place.
        // Finally squeeze data from respective dimensions.
        std::vector<int64_t> dispersed_shape;
        int64_t b_dim_divider = 1;
        for (const auto& el : block_values) {
            b_dim_divider *= el;
        }

        //   note: B_0 is expected to be 1.
        //      x' = reshape(`data`, [B_1, ..., B_{N - 1}, batch / (B_1 * ... B_{N - 1}), D_1, D_2, ...,
        //      D_{N - 1}]),
        //      where B_i = block_shape[i]
        dispersed_shape.insert(dispersed_shape.begin(), block_values.begin() + 1, block_values.end());
        dispersed_shape.push_back(data_shape.at(0) / b_dim_divider);
        for (size_t i = 1; i < data_shape.size(); ++i) {
            dispersed_shape.push_back(data_shape.at(i));
        }

        const auto out_pattern_1 =
                op::Constant::create(element::i64, Shape{dispersed_shape.size()}, dispersed_shape);
        const bool special_zero = false;
        auto flat_node = std::make_shared<ngraph::opset2::Reshape>(data, out_pattern_1, special_zero)
                ->add_provenance_group_members_above({data});

        // calculate axes to transpose
        //      x'' = transpose(x', [N, N + 1, 0, N + 2, 1, ..., N + N - 1, N - 1])
        std::vector<size_t> axes_order{block_values.size() - 1};
        for (size_t i = 0; i < block_values.size() - 1; ++i) {
            axes_order.push_back(i + block_values.size());
            axes_order.push_back(i);
        }
        flat_node = builder::opset1::reorder_axes(flat_node, axes_order);

        //   x''' = reshape(x'', [batch / (B_1 * ... * B_{N - 1}), D_1 * B_1, D_2 * B_2, ... , D_{N - 1}
        //   * B_{N - 1}])
        std::vector<int64_t> squeezed_shape;
        squeezed_shape.push_back(data_shape.at(0) / b_dim_divider);
        for (size_t i = 1; i < block_values.size(); ++i) {
            squeezed_shape.push_back(data_shape.at(i) * block_values.at(i));
        }

        const auto out_pattern_2 =
                op::Constant::create(element::i64, Shape{squeezed_shape.size()}, squeezed_shape);
        flat_node = std::make_shared<opset2::Reshape>(flat_node, out_pattern_2, special_zero)
                ->add_provenance_group_members_above({data});

        //    Crop the start and end of dimensions according to `crops_begin`, `crops_end` to produce
        //    the output of shape:
        //    note: `crops_begin[0], crops_end[0]` are expected to be 0.
        //    `y = [batch / (B_1 * ... * B_{N - 1}), crop(D_1 * B_1, crops_begin[1], crops_end[1]),
        //          crop(D_2 * B_2, crops_begin[2], crops_end[2]), ... ,
        //          crop(D_{N - 1} * B_{N - 1}, crops_begin[N - 1], crops_end[N - 1])]`
        std::vector<int64_t> upperbounds_values;
        auto flat_node_shape = flat_node->get_shape();
        for (size_t i = 0; i < flat_node_shape.size(); ++i) {
            upperbounds_values.push_back(flat_node_shape.at(i) - crops_end_values.at(i));
        }

        const auto upperbounds = opset2::Constant::create(
                crops_end.get_element_type(), Shape{upperbounds_values.size()}, upperbounds_values);

        std::vector<int64_t> begin_mask(data_shape.size(), 0);
        std::vector<int64_t> end_mask(data_shape.size(), 0);
        flat_node = std::make_shared<opset2::StridedSlice>(
                flat_node, crops_begin_const, upperbounds, begin_mask, end_mask);

        flat_node->set_friendly_name(batch_to_space->get_friendly_name());
        ngraph::copy_runtime_info(batch_to_space, flat_node);
        ngraph::replace_node(batch_to_space, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(batch_to_space, "ConvertBatchToSpace");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

void ngraph::pass::ConvertBatchToSpace::convert_batch_to_space_ie_side() {
    auto input0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input1 = ngraph::op::Constant::create(element::i64, Shape{4}, Shape{1, 1, 1, 1});
    auto input2 = ngraph::op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto input3 = ngraph::op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto batch_to_space = std::make_shared<ngraph::opset2::BatchToSpace>(input0, input1, input2, input3);

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto batch_to_space = std::dynamic_pointer_cast<ngraph::opset2::BatchToSpace> (m.get_match_root());
        if (!batch_to_space) {
            return false;
        }

        auto data = batch_to_space->input_value(0);
        auto data_shape = data.get_shape();

        if (transformation_callback(batch_to_space) && (data_shape.size() == 4 || data_shape.size() == 5)) {
            return false;
        }
        auto block = batch_to_space->input_value(1);
        auto crops_begin = batch_to_space->input_value(2);
        auto crops_end = batch_to_space->input_value(3);

        const auto block_const = as_type_ptr<op::Constant>(block.get_node_shared_ptr());
        const auto crops_begin_const = as_type_ptr<op::Constant>(crops_begin.get_node_shared_ptr());
        const auto crops_end_const = as_type_ptr<op::Constant>(crops_end.get_node_shared_ptr());

        std::vector<int64_t> block_values, crops_end_values;
        block_values = block_const->cast_vector<int64_t>();
        crops_end_values = crops_end_const->cast_vector<int64_t>();

        std::vector<int64_t> dispersed_shape(1);
        dispersed_shape.insert(dispersed_shape.end(), data_shape.begin(), data_shape.end());
        std::vector<size_t> axes_order(block_values.size() + 1);
        std::vector<int64_t> squeezed_shape(data_shape.begin(), data_shape.end());
        if (squeezed_shape.size() > block_values.size()) {
            return false;
        }

        NodeVector new_ops;

        std::shared_ptr<Node> flat_node = data.get_node_shared_ptr();
        for (size_t block_idx = 1; block_idx < block_values.size(); ++block_idx) {
            dispersed_shape[0] = block_values[block_idx];
            dispersed_shape[1] /= block_values[block_idx];
            const auto out_pattern_1 =
                    op::Constant::create(element::i64, Shape{dispersed_shape.size()}, dispersed_shape);
            const bool special_zero = false;
            flat_node = std::make_shared<ngraph::op::v1::Reshape>(flat_node, out_pattern_1, special_zero)
                    ->add_provenance_group_members_above({data});
            new_ops.push_back(flat_node);

            size_t val = 1;
            for (size_t axis_idx = 0; axis_idx <= block_values.size(); ++axis_idx) {
                if ((block_idx + 1) == axis_idx) {
                    axes_order[axis_idx] = 0;
                } else {
                    axes_order[axis_idx] = val;
                    val++;
                }
            }

            const auto axes_order_const =
                    ngraph::op::Constant::create(element::i64,
                                         Shape{axes_order.size()},
                                         std::vector<int64_t>(axes_order.begin(), axes_order.end()));
            flat_node = std::make_shared<ngraph::opset1::Transpose>(flat_node, axes_order_const)
                    ->add_provenance_group_members_above({flat_node});
            new_ops.push_back(flat_node);

            squeezed_shape[0] = dispersed_shape[1];
            squeezed_shape[block_idx] *= block_values[block_idx];
            dispersed_shape[block_idx + 1] = squeezed_shape[block_idx];
            const auto out_pattern_2 =
                    op::Constant::create(element::i64, Shape{squeezed_shape.size()}, squeezed_shape);
            flat_node = std::make_shared<ngraph::op::v1::Reshape>(flat_node, out_pattern_2, special_zero)
                    ->add_provenance_group_members_above({data});
            new_ops.push_back(flat_node);
        }

        std::vector<int64_t> upperbounds_values;
        auto flat_node_shape = flat_node->get_shape();
        for (size_t i = 0; i < flat_node_shape.size(); ++i) {
            upperbounds_values.push_back(flat_node_shape.at(i) - crops_end_values.at(i));
        }
        const auto upperbounds = op::Constant::create(
                crops_end.get_element_type(), Shape{upperbounds_values.size()}, upperbounds_values);

        std::vector<int64_t> begin_mask(data_shape.size(), 0);
        std::vector<int64_t> end_mask(data_shape.size(), 0);
        flat_node = std::make_shared<op::v1::StridedSlice>(
                flat_node, crops_begin_const, upperbounds, begin_mask, end_mask);
        new_ops.push_back(flat_node);

        flat_node->set_friendly_name(batch_to_space->get_friendly_name());
        ngraph::copy_runtime_info(batch_to_space, flat_node);
        ngraph::replace_node(batch_to_space, flat_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(batch_to_space, "ConvertBatchToSpace");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}