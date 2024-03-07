// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/gather_compression.hpp"
#include "convert_gather_compression.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <transformations/utils/utils.hpp>
#include "transformations/rt_info/keep_const_precision.hpp"

#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "itt.hpp"

ov::intel_cpu::ConvertToGatherCompression::ConvertToGatherCompression() {
    MATCHER_SCOPE(ConvertToGatherCompression);
    /**
     * Dequantization subgraph pattern
     *
     * Input
     *    \
     *  Convert    zero point
     *      \      /
     *      Subtract    scale
     *         \       /
     *          Multiply
     *             |
     *          Convert
     *             |
     *           Gather
     */
    const element::TypeVector precisions = {ov::element::u8, ov::element::u4};

    auto input_pattern = ov::pass::pattern::wrap_type<opset10::Constant>();
    auto convert1_pattern = ov::pass::pattern::wrap_type<opset10::Convert>({input_pattern}, ov::pass::pattern::consumers_count(1));
    auto zero_point_pattern = ov::pass::pattern::any_input();
    auto subtract_pattern = ov::pass::pattern::wrap_type<opset10::Subtract>({convert1_pattern, zero_point_pattern});
    auto multiply_pattern = ov::pass::pattern::wrap_type<opset10::Multiply>({subtract_pattern, ov::pass::pattern::any_input()});

    auto convert2_pattern = ov::pass::pattern::wrap_type<opset10::Convert>({multiply_pattern}, ov::pass::pattern::consumers_count(1));
    auto check_specific = [](Output<Node> output) -> bool {
        auto node = std::dynamic_pointer_cast<opset10::Constant>(output.get_node_shared_ptr());
        if (node->get_element_type() != ov::element::i32) {
            return false;
        }
        if (node->get_shape() != ov::Shape(1, 1)) {
            return false;
        }
        return node->cast_vector<int32_t>()[0] == 0;
    };
    auto const_pattern = ov::pass::pattern::wrap_type<opset10::Constant>(check_specific);
    auto gather_pattern = ov::pass::pattern::wrap_type<opset8::Gather>({convert2_pattern, ov::pass::pattern::any_input(), const_pattern});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) -> bool {
        const auto& pattern_map = m.get_pattern_value_map();
        auto convert = pattern_map.at(convert1_pattern).get_node_shared_ptr();
        auto input = pattern_map.at(input_pattern);

        const auto gather = m.get_match_root();
        if (transformation_callback(gather)) {
            return false;
        }

        // Check input_precision
        const auto& input_precision = input.get_element_type();
        if (std::find(precisions.begin(), precisions.end(), input_precision) == precisions.end()) {
            return false;
        }
        if (input.get_shape().size() != 2u) {
            return false;
        }

        // Subtract
        std::shared_ptr<ov::Node> zp = nullptr;
        auto zero_point = pattern_map.at(zero_point_pattern).get_node_shared_ptr();
        if (ov::is_type<opset10::Constant>(zero_point)) {
            if (input_precision != zero_point->get_input_element_type(0)) {
                return false;
            }
            zp = zero_point;
        } else if (ov::is_type<opset10::Convert>(zero_point) &&
                   input_precision == zero_point->get_input_element_type(0) &&
                   ov::is_type<opset10::Constant>(zero_point->get_input_node_ptr(0))) {
            zp = zero_point->get_input_node_shared_ptr(0);
        } else {
            return false;
        }
        // Shape=[?, 1]
        if (!(zp->get_shape().size() == 2 && zp->get_shape()[1] == 1u)) {
            return false;
        }
        auto input_zp =
            zp->get_element_type() != ov::element::f32 ? std::make_shared<opset10::Convert>(zp, ov::element::f32) : zp;

        // Scale
        std::shared_ptr<ov::Node> scale = nullptr;
        auto scale_convert = pattern_map.at(multiply_pattern).get_node_shared_ptr()->get_input_node_shared_ptr(1);
        if (ov::is_type<opset10::Convert>(scale_convert) &&
            ov::is_type<opset10::Constant>(scale_convert->get_input_node_shared_ptr(0))) {
            scale = scale_convert->get_input_node_shared_ptr(0);
        } else if (ov::is_type<opset10::Constant>(scale_convert)) {
            scale = scale_convert;
        } else {
            return false;
        }
        // Shape=[?, 1]
        if (!(scale->get_shape().size() == 2 && scale->get_shape()[1] == 1u)) {
            return false;
        }
        auto input_scale = scale->get_element_type() != ov::element::f32
                               ? std::make_shared<opset10::Convert>(scale, ov::element::f32)
                               : scale;

        // Replace gather with GatherCompressionNode
        const auto gatherCompression = std::make_shared<GatherCompressionNode>(input,
                                                                               input_zp,
                                                                               input_scale,
                                                                               gather->get_input_node_shared_ptr(1),
                                                                               gather->output(0).get_element_type());

        gatherCompression->set_friendly_name(gather->get_friendly_name());
        ov::copy_runtime_info(gather, gatherCompression);
        ov::replace_node(gather, gatherCompression);

        // It is necessary to avoid precision conversion for constant node(compressed weights)
        ov::enable_keep_const_precision(gatherCompression->get_input_node_shared_ptr(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather_pattern, matcher_name);
    this->register_matcher(m, callback);
}
