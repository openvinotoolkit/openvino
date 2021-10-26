// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <legacy/ngraph_ops/crop_ie.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertStridedSliceToCropMatcher, "ConvertStridedSliceToCropMatcher", 0);

ngraph::pass::ConvertStridedSliceToCropMatcher::ConvertStridedSliceToCropMatcher() {
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto m_begin = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
    auto m_end = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
    auto m_stride = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
    std::vector<int64_t> begin_mask = {0, 0, 0, 0};
    std::vector<int64_t> end_mask = {0, 0, 0, 0};
    auto m_slice = std::make_shared<ngraph::opset1::StridedSlice>(data, m_begin, m_end, m_stride, begin_mask, end_mask);

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto slice = std::dynamic_pointer_cast<ngraph::opset1::StridedSlice> (m.get_match_root());
        if (!slice || transformation_callback(slice)) {
            return false;
        }

        auto data_output = slice->input_value(0);
        auto begin_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(slice->input_value(1).get_node_shared_ptr());
        auto end_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(slice->input_value(2).get_node_shared_ptr());
        auto stride_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(slice->input_value(3).get_node_shared_ptr());

        auto partial_input_shape = slice->get_input_partial_shape(0);

        if (!begin_node || !end_node || !stride_node || partial_input_shape.is_dynamic()) {
            return false;
        }

        auto input_shape = slice->get_input_shape(0);
        auto output_shape = slice->get_output_shape(0);

        auto begin = begin_node->cast_vector<int64_t>();
        auto end = end_node->cast_vector<int64_t>();
        auto strides = stride_node->cast_vector<int64_t>();

        bool ones_stride = true;
        for (auto & s : strides) {
            if (s != 1) ones_stride = false;
        }

        if (!ones_stride) return false;

        auto convert_to_set = [](const std::vector<int64_t> mask) {
            AxisSet axis_set{};
            for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i) {
                if (mask[i] == 1) {
                    axis_set.emplace(i);
                }
            }
            return axis_set;
        };

        auto shrink_axis_mask = convert_to_set(slice->get_shrink_axis_mask());
        auto new_axis_mask = convert_to_set(slice->get_new_axis_mask());
        auto ellipsis_mask = convert_to_set(slice->get_ellipsis_mask());
        auto begin_mask = convert_to_set(slice->get_begin_mask());
        auto end_mask = convert_to_set(slice->get_end_mask());

        std::vector<int64_t> reshape_pattern,
                axes,
                offset,
                dim;

        size_t input_shape_idx = 0;
        uint64_t uniq_id = 0;
        for (size_t axis = 0; axis < begin.size(); ++axis) {
            // add dimensions hidden under the ellipsis mask if ellipsis mask is set
            if (ellipsis_mask.count(axis)) {
                // only one bit in ellipsis mask is allowed
                int num_new_axis_after_ellipses = 0;
                int num_input_axis_before_ellipses = 0;
                for (size_t i = 0; i < axis; ++i) {
                    if (!new_axis_mask.count(i))
                        num_input_axis_before_ellipses++;
                }
                for (size_t i = axis + 1; i < begin.size(); ++i) {
                    if (new_axis_mask.count(i))
                        num_new_axis_after_ellipses++;
                }

                // -1 because it's a position of ellipses
                size_t num_input_axis_after_ellipses = (begin.size() - axis - num_new_axis_after_ellipses - 1);
                size_t num_of_hidden_dims = input_shape.size() - num_input_axis_after_ellipses
                                                   - num_input_axis_before_ellipses;
                for (size_t i = 0; i < num_of_hidden_dims; ++i) {
                    axes.emplace_back(uniq_id);
                    uniq_id++;
                    reshape_pattern.emplace_back(input_shape[input_shape_idx]);
                    offset.emplace_back(0);

                    dim.emplace_back(input_shape[input_shape_idx]);
                    input_shape_idx++;
                }
            } else {
                // add new single dimension if new_axis_mask is set
                if (new_axis_mask.count(axis)) {
                    reshape_pattern.emplace_back(1);
                    dim.emplace_back(1);
                    offset.emplace_back(0);
                } else if (shrink_axis_mask.count(axis)) {
                    // skip this dimension if shrink_axis_mask is set (input_shape_idx++)
                    dim.emplace_back(1);
                    offset.emplace_back(begin_mask.count(axis) ? 0 : begin[axis]);
                    reshape_pattern.emplace_back(1);
                    input_shape_idx++;
                } else {
                    // calculate dimension using begin, end, begin_mask, end_mask, stride
                    reshape_pattern.emplace_back(input_shape[input_shape_idx]);

                    int64_t lb = begin[axis];
                    int64_t ub = end[axis];

                    // convert negative indexes to positive
                    if (lb < 0)
                        lb = std::max(static_cast<int64_t>(input_shape[input_shape_idx]) + lb,
                                      static_cast<int64_t>(0));
                    if (ub < 0)
                        ub = std::max(static_cast<int64_t>(input_shape[input_shape_idx]) + ub,
                                      static_cast<int64_t>(0));

                    // apply restrictions when begin or end values more/less than max/min possible values.
                    lb = std::min(static_cast<int64_t>(input_shape[input_shape_idx]), lb);
                    ub = std::min(static_cast<int64_t>(input_shape[input_shape_idx]), ub);


                    // set default value for stride or use given value
                    int64_t stride = 1;
                    if (strides.size() > axis)
                        stride = strides[axis];

                    int64_t dimension = 0;
                    if (stride < 0) {
                        // apply masks
                        if (begin_mask.count(axis))
                            lb = static_cast<int64_t>(input_shape[input_shape_idx]) - 1;
                        if (end_mask.count(axis))
                            ub = -1;

                        lb = std::min(lb, static_cast<int64_t>(input_shape[input_shape_idx]) - 1);
                        offset.emplace_back(lb);
                        lb -= 1;  // we always get 1st element, so we need decrease range
                        if (ub <= lb)
                            dimension = (ub - lb) / stride + 1;
                    } else {
                        // apply masks
                        if (begin_mask.count(axis))
                            lb = 0;
                        offset.emplace_back(lb);

                        if (end_mask.count(axis)) {
                            ub = static_cast<int64_t>(input_shape[input_shape_idx]);
                        }

                        lb += 1;  // we always get 1st element, so we need decrease range
                        if (ub >= lb) {
                            dimension = (ub - lb) / stride + 1;
                        }
                    }

                    dim.emplace_back(dimension);
                    input_shape_idx++;
                }
                axes.emplace_back(uniq_id);
                uniq_id++;
            }
        }
        for (; input_shape_idx < input_shape.size(); ++input_shape_idx) {
            reshape_pattern.emplace_back(input_shape[input_shape_idx]);
            offset.emplace_back(0);
            dim.emplace_back(input_shape[input_shape_idx]);
            axes.emplace_back(uniq_id);
            uniq_id++;
        }

        // CLDNN: if (cropLayer->axis[i] < 0 || cropLayer->axis[i] > 3) -> invalid crop axis
        if (axes.size() > 4) {
            return false;
        }

        NodeVector new_ops;

        // Reshape in case of new axis
        if (!new_axis_mask.empty()) {
            auto new_shape = std::make_shared<ngraph::opset1::Constant>(element::i64,
                                                                    ngraph::Shape{reshape_pattern.size()}, reshape_pattern);
            auto data_node = std::make_shared<ngraph::opset1::Reshape>(data_output, new_shape, true);
            data_node->set_friendly_name(slice->get_friendly_name() + "/Reshape_before");
            new_ops.push_back(data_node);
            data_output = data_node->output(0);
        }

        auto data_node_shape = data_output.get_shape();
        // MKLDNN: "Crop supports only 2d, 4d and 5d blobs."
        if (data_node_shape.size() != 2 && data_node_shape.size() != 4 && data_node_shape.size() != 5) {
            return false;
        }

        // Crop
        std::shared_ptr<ngraph::Node> data_node = std::make_shared<ngraph::op::CropIE> (data_output, axes, dim, offset);
        data_node->set_friendly_name(slice->get_friendly_name());
        new_ops.push_back(data_node);

        auto crop_data_node = data_node;

        // Reshape in case of deleting of axis
        if (!shrink_axis_mask.empty()) {
            auto new_shape = std::make_shared<ngraph::opset1::Constant>(element::i64, ngraph::Shape{output_shape.size()},
                                                                    output_shape);
            data_node = std::make_shared<ngraph::opset1::Reshape>(data_node->output(0), new_shape, true);
            crop_data_node->set_friendly_name(slice->get_friendly_name() + "/Crop");
            data_node->set_friendly_name(slice->get_friendly_name());
            new_ops.push_back(data_node);
        }

        ngraph::copy_runtime_info(slice, new_ops);
        ngraph::replace_node(slice, data_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_slice, "ConvertStridedSliceToCrop");
    this->register_matcher(m, callback);
}
