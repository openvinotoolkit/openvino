// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_group_deconv_1d_to_2d.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ConvertGroupConvBackpropData1DTo2D::ConvertGroupConvBackpropData1DTo2D() {
    using namespace ov::pass::pattern;

    auto deconv_m = wrap_type<ov::op::v1::GroupConvolutionBackpropData>([](const ov::Output<ov::Node>& output) {
        const auto& node = output.get_node_shared_ptr();
        return node->get_input_partial_shape(0).rank().is_static() && node->get_input_partial_shape(0).rank().get_length() == 3 &&
               node->get_input_partial_shape(1).rank().is_static() && node->get_input_partial_shape(1).rank().get_length() == 4;
    });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto deconv = ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(m.get_match_root());
        if (!deconv || transformation_callback(deconv))
            return false;

        const auto last_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto data = std::make_shared<ov::op::v0::Unsqueeze>(deconv->input_value(0), last_axis);
        auto weights = std::make_shared<ov::op::v0::Unsqueeze>(deconv->input_value(1), last_axis);

        auto strides = deconv->get_strides();
        auto dilations = deconv->get_dilations();
        auto pads_begin = deconv->get_pads_begin();
        auto pads_end = deconv->get_pads_end();
        auto output_padding = deconv->get_output_padding();
        strides.push_back(1);
        dilations.push_back(1);
        pads_begin.push_back(0);
        pads_end.push_back(0);
        if (!output_padding.empty())
            output_padding.push_back(0);

        ov::NodeVector new_ops{data, weights};
        std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData> deconv_2d;
        if (deconv->get_input_size() == 3) {
            const auto& output_shape = deconv->input_value(2);
            auto unit = ov::op::v0::Constant::create(output_shape.get_element_type(), ov::Shape{1}, {1});
            auto output_shape_2d = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{output_shape, unit}, 0);
            new_ops.push_back(output_shape_2d);
            deconv_2d = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data,
                                                                                   weights,
                                                                                   output_shape_2d,
                                                                                   strides,
                                                                                   pads_begin,
                                                                                   pads_end,
                                                                                   dilations,
                                                                                   deconv->get_auto_pad(),
                                                                                   output_padding);
        } else {
            deconv_2d = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data,
                                                                                   weights,
                                                                                   strides,
                                                                                   pads_begin,
                                                                                   pads_end,
                                                                                   dilations,
                                                                                   deconv->get_auto_pad(),
                                                                                   output_padding);
        }
        auto output = std::make_shared<ov::op::v0::Squeeze>(deconv_2d, last_axis);
        new_ops.push_back(deconv_2d);
        new_ops.push_back(output);

        output->set_friendly_name(deconv->get_friendly_name());
        ov::copy_runtime_info(deconv, new_ops);
        ov::replace_node(deconv, output);
        return true;
    };

    auto m = std::make_shared<Matcher>(deconv_m, "ConvertGroupConvBackpropData1DTo2D");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
