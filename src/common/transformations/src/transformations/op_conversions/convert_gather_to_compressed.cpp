// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_to_compressed.hpp"

#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertGatherToGatherCompressed::ConvertGatherToGatherCompressed() {
    using namespace ov::pass::pattern;

    auto compressed_constant = [](const ov::Output<ov::Node>& output) {
        return (output.get_element_type() == ov::element::u8 || output.get_element_type() == ov::element::i8 ||
                output.get_element_type() == ov::element::u4 || output.get_element_type() == ov::element::i4) &&
               output.get_target_inputs().size() == 1 &&
               (output.get_shape().size() == 2 || output.get_shape().size() == 3);
    };

    auto reshape_3d_to_2d = [](const ov::Output<ov::Node>& output) {
        auto in_ps = output.get_node()->get_input_partial_shape(0);
        auto out_ps = output.get_node()->get_output_partial_shape(0);
        return in_ps.rank().is_static() && out_ps.rank().is_static() && in_ps.size() == 3 && out_ps.size() == 2;
    };

    auto dicts_m = wrap_type<ov::op::v0::Constant>(compressed_constant);
    auto convert_m = wrap_type<ov::op::v0::Convert>({dicts_m});

    auto sub_const_m = ov::pass::pattern::any_input();  // const or const+convert
    auto subtract_m = wrap_type<ov::op::v1::Subtract>({convert_m, sub_const_m});

    auto mul_const_m = ov::pass::pattern::any_input();  // const or const+convert
    auto mul_with_sub_m = wrap_type<ov::op::v1::Multiply>({subtract_m, mul_const_m});
    auto mul_no_sub_m = wrap_type<ov::op::v1::Multiply>({convert_m, mul_const_m});
    auto mul_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{mul_with_sub_m, mul_no_sub_m});

    auto reshape_const_m = wrap_type<ov::op::v0::Constant>();
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({mul_m, reshape_const_m}, reshape_3d_to_2d);

    auto last_convert_input = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{reshape_m, mul_m});
    auto last_convert_m = wrap_type<ov::opset10::Convert>({last_convert_input});

    auto dicts_input_m =
        std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{reshape_m, last_convert_m, mul_m});
    auto gather_m = wrap_type<ov::opset10::Gather>({dicts_input_m, any_input(), wrap_type<ov::op::v0::Constant>()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(gather_m));
        OPENVINO_ASSERT(pattern_map.count(mul_const_m));
        OPENVINO_ASSERT(pattern_map.count(dicts_m));
        OPENVINO_ASSERT(pattern_map.count(convert_m));
        ov::Shape dicts_shape = pattern_map.at(dicts_m).get_node_shared_ptr()->get_shape();
        auto gather_node =
            std::dynamic_pointer_cast<ov::opset10::Gather>(pattern_map.at(gather_m).get_node_shared_ptr());
        if (!gather_node || transformation_callback(gather_node)) {
            return false;
        }

        auto reshape_const_to_2d = [](std::shared_ptr<ov::Node> node) -> std::shared_ptr<ov::Node> {
            auto convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(node);
            if (convert != nullptr) {
                return node;
            } else if (std::dynamic_pointer_cast<ov::op::v1::Reshape>(node) != nullptr) {
                return node;
            } else {
                auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
                OPENVINO_ASSERT(constant != nullptr);
                ov::Shape current_shape = constant->get_shape();
                if (current_shape.size() <= 2)
                    return constant;
                OPENVINO_ASSERT(current_shape.size() == 3);
                auto new_shape = ov::Shape{current_shape[0], current_shape[1] * current_shape[2]};
                return std::make_shared<ov::op::v0::Constant>(*constant, new_shape);
            }
        };

        bool reshape_to_2d = (pattern_map.count(reshape_m) > 0) ? true : false;

        std::shared_ptr<ov::Node> gather_input_a =
            reshape_to_2d ? reshape_const_to_2d(pattern_map.at(dicts_m).get_node_shared_ptr())
                          : pattern_map.at(dicts_m).get_node_shared_ptr();
        const auto& gather_input_b = gather_node->get_input_node_shared_ptr(1);
        const auto& gather_input_c = gather_node->get_input_node_shared_ptr(2);
        const auto& scale = reshape_to_2d ? reshape_const_to_2d(pattern_map.at(mul_const_m).get_node_shared_ptr())
                                          : pattern_map.at(mul_const_m).get_node_shared_ptr();
        std::shared_ptr<ov::Node> optional_zero_point = nullptr;

        const bool with_zero_point = pattern_map.count(subtract_m) > 0;
        if (with_zero_point) {
            optional_zero_point = reshape_to_2d ? reshape_const_to_2d(pattern_map.at(sub_const_m).get_node_shared_ptr())
                                                : pattern_map.at(sub_const_m).get_node_shared_ptr();
        }

        std::shared_ptr<ov::Node> gather_input_scale = scale;
        std::shared_ptr<ov::Node> gather_input_zp = optional_zero_point;
        std::vector<std::shared_ptr<ov::Node>> result_nodes = {};

        std::shared_ptr<ov::Node> new_gather_node = nullptr;
        if (with_zero_point) {
            new_gather_node =
                std::make_shared<ov::op::internal::GatherCompressed>(gather_input_a,
                                                                     gather_input_b,
                                                                     gather_input_c,
                                                                     gather_node->get_batch_dims(),
                                                                     gather_input_scale,
                                                                     gather_input_zp,
                                                                     gather_node->get_output_element_type(0));
        } else {
            new_gather_node =
                std::make_shared<ov::op::internal::GatherCompressed>(gather_input_a,
                                                                     gather_input_b,
                                                                     gather_input_c,
                                                                     gather_node->get_batch_dims(),
                                                                     gather_input_scale,
                                                                     gather_node->get_output_element_type(0));
        }

        transformation_callback(new_gather_node);

        result_nodes.push_back(new_gather_node);
        new_gather_node->set_friendly_name(gather_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(gather_node, new_gather_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather_m, "ConvertGatherToGatherCompressed");
    this->register_matcher(m, callback);
}
