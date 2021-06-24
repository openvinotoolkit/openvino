// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertDeformableConv8To1, "ConvertDeformableConv8To1", 0);

ngraph::pass::ConvertDeformableConv8To1::ConvertDeformableConv8To1() {
    MATCHER_SCOPE(ConvertDeformableConv8To1);

    auto deformable_conv_v8 = pattern::wrap_type<ngraph::opset8::DeformableConvolution>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto deformable_conv_v8_node = std::dynamic_pointer_cast<ngraph::opset8::DeformableConvolution>(m.get_match_root());
        if (!deformable_conv_v8_node)
            return false;

        if (deformable_conv_v8_node->get_input_size() != 3
            || deformable_conv_v8_node->get_use_bilinear_interpolation_padding() != false)
            return false;

        auto arg = deformable_conv_v8_node->input_value(0);
        auto offsets = deformable_conv_v8_node->input_value(1);
        auto filters = deformable_conv_v8_node->input_value(2);

        auto deformable_conv_v1 =
                std::make_shared<ngraph::opset1::DeformableConvolution>(arg,
                                                                        offsets,
                                                                        filters,
                                                                        deformable_conv_v8_node->get_strides(),
                                                                        deformable_conv_v8_node->get_pads_begin(),
                                                                        deformable_conv_v8_node->get_pads_end(),
                                                                        deformable_conv_v8_node->get_dilations(),
                                                                        deformable_conv_v8_node->get_auto_pad(),
                                                                        deformable_conv_v8_node->get_group(),
                                                                        deformable_conv_v8_node->get_deformable_group());
        deformable_conv_v1->set_friendly_name(deformable_conv_v8_node->get_friendly_name());
        ngraph::copy_runtime_info(deformable_conv_v8_node, deformable_conv_v1);
        ngraph::replace_node(deformable_conv_v8_node, deformable_conv_v1);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(deformable_conv_v8, matcher_name);
    register_matcher(m, callback);
}
