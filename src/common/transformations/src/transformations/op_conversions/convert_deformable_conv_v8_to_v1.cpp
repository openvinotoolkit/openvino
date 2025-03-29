// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertDeformableConv8To1::ConvertDeformableConv8To1() {
    MATCHER_SCOPE(ConvertDeformableConv8To1);

    auto deformable_conv_v8 = pattern::wrap_type<ov::op::v8::DeformableConvolution>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto deformable_conv_v8_node = ov::as_type_ptr<ov::op::v8::DeformableConvolution>(m.get_match_root());
        if (!deformable_conv_v8_node)
            return false;

        if (deformable_conv_v8_node->get_input_size() != 3 || deformable_conv_v8_node->get_bilinear_interpolation_pad())
            return false;

        auto arg = deformable_conv_v8_node->input_value(0);
        auto offsets = deformable_conv_v8_node->input_value(1);
        auto filters = deformable_conv_v8_node->input_value(2);

        auto deformable_conv_v1 =
            std::make_shared<ov::op::v1::DeformableConvolution>(arg,
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
        ov::copy_runtime_info(deformable_conv_v8_node, deformable_conv_v1);
        ov::replace_node(deformable_conv_v8_node, deformable_conv_v1);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(deformable_conv_v8, matcher_name);
    register_matcher(m, callback);
}
