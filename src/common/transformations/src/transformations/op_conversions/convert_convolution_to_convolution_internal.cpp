// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convolution_to_convolution_internal.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/convolution.hpp"

ov::pass::ConvertConvolutionToConvolutionInternal::ConvertConvolutionToConvolutionInternal() {
    MATCHER_SCOPE(ConvertConvolutionToConvolutionInternal);

    auto Convolution = pattern::wrap_type<ov::op::v1::Convolution>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(m.get_match_root());
        if (!conv) {
            return false;
        }

        auto new_conv = std::make_shared<ov::op::internal::Convolution>(conv->input_value(0),
                                                                        conv->input_value(1),
                                                                        Output<Node>(),
                                                                        conv->get_strides(),
                                                                        conv->get_pads_begin(),
                                                                        conv->get_pads_end(),
                                                                        conv->get_dilations(),
                                                                        -1,
                                                                        conv->get_auto_pad(),
                                                                        conv->get_output_element_type(0));

        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info(conv, new_conv);
        ov::replace_node(conv, new_conv);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(Convolution, matcher_name);
    this->register_matcher(m, callback);
}
