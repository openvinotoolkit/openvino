// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::NearestNeighborUpsamplingFusion, "NearestNeighborUpsamplingFusion", 0);

ngraph::pass::NearestNeighborUpsamplingFusion::NearestNeighborUpsamplingFusion() {
    MATCHER_SCOPE(NearestNeighborUpsamplingFusion);
    auto input = ngraph::pattern::any_input();
    auto shape_of = std::make_shared<ngraph::opset8::ShapeOf>(input);

    auto slice_begin = pattern::wrap_type<ngraph::opset8::Constant>();
    auto slice_end = pattern::wrap_type<ngraph::opset8::Constant>();
    auto slice_stride = pattern::wrap_type<ngraph::opset8::Constant>();
    auto strided_slice = ngraph::pattern::wrap_type<ngraph::opset8::StridedSlice>({shape_of, slice_begin, slice_end, slice_stride});

    auto unsqueeze_1_axis = pattern::wrap_type<ngraph::opset8::Constant>();
    auto unsqueeze_1 = ngraph::pattern::wrap_type<ngraph::opset8::Unsqueeze>({strided_slice, unsqueeze_1_axis});

    auto unsqueeze_2_axis = pattern::wrap_type<ngraph::opset8::Constant>();
    auto unsqueeze_2 = ngraph::pattern::wrap_type<ngraph::opset8::Unsqueeze>({strided_slice, unsqueeze_2_axis});

    auto concat_1_h = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_1_one_1 = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_1_w = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_1_one_2 = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_1_c = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_1 = pattern::wrap_type<ngraph::opset8::Concat>({unsqueeze_1, concat_1_h, concat_1_one_1, concat_1_w, concat_1_one_2, concat_1_c});

    auto concat_2_h = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_2_w = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_2_c = pattern::wrap_type<ngraph::opset8::Constant>();
    auto concat_2 = pattern::wrap_type<ngraph::opset8::Concat>({unsqueeze_2, concat_2_h, concat_2_w, concat_2_c});

    auto reshape_1 = pattern::wrap_type<ngraph::opset8::Reshape>({input, concat_1});

    auto mul_const = pattern::wrap_type<ngraph::opset8::Constant>();
    auto mul = pattern::wrap_type<ngraph::opset8::Multiply>({reshape_1, mul_const});

    auto reshape_2 = pattern::wrap_type<ngraph::opset8::Reshape>({mul, concat_2});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto &pattern_to_output = m.get_pattern_value_map();

        const auto reshape_2_node = std::dynamic_pointer_cast<ngraph::opset8::Reshape>(pattern_to_output.at(reshape_2).get_node_shared_ptr());
        const auto mul_node = std::dynamic_pointer_cast<ngraph::opset8::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr());
        const auto mul_const_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(mul_const).get_node_shared_ptr());

        if (!reshape_2_node || !mul_node || !mul_const_node) return false;

        const auto mul_const_value = mul_const_node->cast_vector<float>();
        if (mul_const_value != std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}) return false;
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape_2, matcher_name);
    register_matcher(m, callback);
}
