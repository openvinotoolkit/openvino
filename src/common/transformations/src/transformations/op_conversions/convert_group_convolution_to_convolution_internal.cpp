// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_group_convolution_to_convolution_internal.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/convolution.hpp"

ov::pass::ConvertGroupConvolutionToConvolutionInternal::ConvertGroupConvolutionToConvolutionInternal() {
    MATCHER_SCOPE(ConvertGroupConvolutionToConvolutionInternal);

    auto GroupConvolution = pattern::wrap_type<ov::op::v1::GroupConvolution>();

    matcher_pass_callback gconv_callback = [](pattern::Matcher& m) {
        auto gconv = ov::as_type_ptr<ov::op::v1::GroupConvolution>(m.get_match_root());
        if (!gconv) {
            return false;
        }

        int64_t groups = -1;
        auto weights_shape = gconv->get_input_partial_shape(1);
        if (weights_shape.rank().is_dynamic() || weights_shape[0].is_dynamic()) {
            return false;
        }
        groups = weights_shape[0].get_length();

        auto new_conv = std::make_shared<ov::op::internal::Convolution>(gconv->input_value(0),
                                                                        gconv->input_value(1),
                                                                        Output<Node>(),
                                                                        gconv->get_strides(),
                                                                        gconv->get_pads_begin(),
                                                                        gconv->get_pads_end(),
                                                                        gconv->get_dilations(),
                                                                        groups,
                                                                        gconv->get_auto_pad(),
                                                                        gconv->get_output_element_type(0));

        new_conv->set_friendly_name(gconv->get_friendly_name());
        copy_runtime_info(gconv, new_conv);
        replace_node(gconv, new_conv);
        return true;
    };

    auto gconv_m = std::make_shared<pattern::Matcher>(GroupConvolution, matcher_name);
    this->register_matcher(gconv_m, gconv_callback);
}
