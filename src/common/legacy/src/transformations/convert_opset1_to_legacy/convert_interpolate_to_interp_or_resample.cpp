// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_interpolate_to_interp_or_resample.hpp"

#include <memory>
#include <vector>
#include <string>
#include <set>
#include <map>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <legacy/ngraph_ops/interp.hpp>

ngraph::pass::ConvertInterpolateToInterpOrResampleMatcher::ConvertInterpolateToInterpOrResampleMatcher() {
    auto interpolate = pattern::wrap_type<opset1::Interpolate>({pattern::any_input(pattern::has_static_shape()),
                                                                pattern::wrap_type<opset1::Constant>()});

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto interpolate = std::dynamic_pointer_cast<ngraph::opset1::Interpolate> (m.get_match_root());
        if (!interpolate)
            return false;

        auto data_node = interpolate->input_value(0);
        auto out_shape_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(interpolate->input_value(1).get_node_shared_ptr());
        auto interpolate_attrs = interpolate->get_attrs();
        auto input_shape = data_node.get_shape();

        if (!out_shape_node) {
            return false;
        }

        auto out_spatial_shape = out_shape_node->cast_vector<int64_t> ();

        std::size_t num_of_spatial_vars = input_shape.size() - 2;
        auto interpolate_axes = interpolate_attrs.axes;
        auto interpolate_mode = interpolate_attrs.mode;
        if (num_of_spatial_vars != 2 && num_of_spatial_vars != 3) {
            return false;
        }

        if (interpolate_attrs.pads_begin.empty())
            interpolate_attrs.pads_begin = std::vector<size_t>{0};
        if (interpolate_attrs.pads_end.empty())
            interpolate_attrs.pads_end =  std::vector<size_t>{0};

        std::vector<size_t> useless_axes;
        size_t axis_idx = 0;
        for (size_t axis = 0; axis < input_shape.size(); ++axis) {
            if (interpolate_axes.count(axis)) {
                if (static_cast<int64_t>(input_shape[axis]) == out_spatial_shape[axis_idx] && axis < 2)
                    // keeping only those not spatial dimensions that are going to be changed
                    useless_axes.push_back(axis);
                ++axis_idx;
            }
        }

        std::reverse(useless_axes.begin(), useless_axes.end());
        for (const auto & axis : useless_axes) {
            interpolate_axes.erase(axis);
            out_spatial_shape.erase(out_spatial_shape.begin() + axis);
        }

        // Interpolate can be converted when interpolation is performed over spatial dimensions only
        if (num_of_spatial_vars == 2 && interpolate_axes != AxisSet{2, 3}) {
            return false;
        }

        if (num_of_spatial_vars == 2 && interpolate_axes.size() == 2 && std::set<std::string>{"nearest", "cubic", "area"}.count(interpolate_mode) == 0) {
            auto attrs = ngraph::op::InterpolateIEAttrs();
            attrs.pad_beg = static_cast<int>(interpolate_attrs.pads_begin[0]);
            attrs.pad_end = static_cast<int>(interpolate_attrs.pads_end[0]);
            attrs.height = static_cast<int>(out_spatial_shape[0]);
            attrs.width = static_cast<int>(out_spatial_shape[1]);
            attrs.align_corners = interpolate_attrs.align_corners;
            attrs.mode = interpolate_mode;
            attrs.antialias = interpolate_attrs.antialias;

            auto interp = std::make_shared<ngraph::op::Interp>(data_node, attrs);
            interp->set_friendly_name(interpolate->get_friendly_name());

            ngraph::copy_runtime_info(interpolate, interp);
            ngraph::replace_node(interpolate, interp);
        } else if (interpolate_attrs.pads_begin[0] == 0 && interpolate_attrs.pads_end[0] == 0 && !interpolate_attrs.align_corners) {
            auto attrs = ngraph::op::ResampleIEAttrs();
            attrs.mode = interpolate_mode;
            attrs.antialias = interpolate_attrs.antialias;

            std::shared_ptr<Node> resample;

            if (num_of_spatial_vars == 3 && interpolate_axes != AxisSet{2, 3, 4}) {
                auto corrected_output_spatial_shape = std::vector<int64_t>(num_of_spatial_vars);

                std::map<int64_t, int64_t> axis_to_index;
                /*
                 * Because interpolate_axes != AxisSet{2, 3, 4} and num_of_spatial_vars == 3, then
                 * interpolate_axes can have one of the following values:
                 *      {2}, {3}, {4}, {2, 3}, {2, 4}
                 * Sizes of out_spatial_shape are correspondigly 1, 1, 1, 2, 2.
                 * Hence, we need to add missing axes, and shape component for missing axes will be taken from input_shape.
                 */
                int64_t counter = 0;
                for (int64_t axis : interpolate_axes) {
                    axis_to_index[axis] = counter++;
                }

                for (int64_t axis = 2; axis <= 4; ++axis) {
                    auto it = interpolate_axes.find(axis);
                    if (it != interpolate_axes.end()) {
                        corrected_output_spatial_shape[axis - 2] = out_spatial_shape[axis_to_index[axis]];
                    } else {
                        corrected_output_spatial_shape[axis - 2] = input_shape[axis];
                    }
                }

                out_spatial_shape = corrected_output_spatial_shape;
            }

            // In case if output shape differ only in spatial dims and can be produced by using factor we set factor attr
            bool has_same_factor(true);
            int64_t factor(0);
            for (size_t i = 0; i < out_spatial_shape.size(); ++i) {
                if (out_spatial_shape[i] % input_shape[i + 2] == 0) {
                    int64_t f = out_spatial_shape[i] / input_shape[i + 2];
                    if (factor == 0) {
                        factor = f;
                    } else if (factor != f) {
                        has_same_factor = false;
                    }
                } else {
                    has_same_factor = false;
                }
            }

            if (has_same_factor && factor != 0) {
                attrs.factor = factor;
                resample = std::make_shared<ngraph::op::ResampleV2>(data_node, attrs);
            } else {
                // first concatenates [N,C] shapes from the input tensor with the Interpolate second input value to
                // create the desired output shape for the Resample
                auto output_shape = out_spatial_shape;
                output_shape.insert(output_shape.begin(), input_shape[0]);
                output_shape.insert(output_shape.begin() + 1, input_shape[1]);
                auto constant = std::make_shared<ngraph::opset1::Constant>(out_shape_node->get_element_type(), Shape{output_shape.size()}, output_shape);
                resample = std::make_shared<ngraph::op::ResampleV2>(data_node, constant, attrs);
            }

            resample->set_friendly_name(interpolate->get_friendly_name());
            ngraph::copy_runtime_info(interpolate, resample);
            ngraph::replace_node(interpolate, resample);
        } else {
            return false;
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interpolate, "ConvertInterpolateToInterpOrResample");
    this->register_matcher(m, callback);
}
