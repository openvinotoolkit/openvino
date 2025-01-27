// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fake_convert_decomposition.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::FakeConvertDecomposition::FakeConvertDecomposition() {
    MATCHER_SCOPE(FakeConvertDecomposition);
    auto data = pattern::any_input();

    auto fake_convert = ov::pass::pattern::wrap_type<ov::op::v13::FakeConvert>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        const auto fake_convert_node =
            ov::as_type_ptr<ov::op::v13::FakeConvert>(pattern_to_output.at(fake_convert).get_node_shared_ptr());

        if (fake_convert_node == nullptr || transformation_callback(fake_convert_node)) {
            return false;
        }

        Output<Node> data{fake_convert_node->input_value(0)};
        const Output<Node> input_scale{fake_convert_node->input_value(1)};
        auto input_type = data.get_element_type();

        ov::pass::NodeRegistry decomp_ops;
        if (input_type != input_scale.get_element_type()) {
            input_type = input_scale.get_element_type();
            data = std::make_shared<ov::op::v0::Convert>(data, input_type);
            data = decomp_ops.add(data.get_node_shared_ptr());
        }

        // Align with clamp behavior of FakeConvert in ngraph reference
        const auto lower_bound = fake_convert_node->get_destination_element_type() == ov::element::f8e4m3
                                     ? static_cast<float>(std::numeric_limits<ov::float8_e4m3>::lowest())
                                     : static_cast<float>(std::numeric_limits<ov::float8_e5m2>::lowest());
        const auto upper_bound = fake_convert_node->get_destination_element_type() == ov::element::f8e4m3
                                     ? static_cast<float>(std::numeric_limits<ov::float8_e4m3>::max())
                                     : static_cast<float>(std::numeric_limits<ov::float8_e5m2>::max());

        std::shared_ptr<Node> result;
        const auto scale = decomp_ops.make<ov::op::v1::Multiply>(data, input_scale);
        if (fake_convert_node->get_input_size() == 2) {
            const auto clamp = decomp_ops.make<ov::op::v0::Clamp>(scale, lower_bound, upper_bound);
            const auto downconvert =
                decomp_ops.make<ov::op::v0::Convert>(clamp, fake_convert_node->get_destination_element_type());
            const auto upconvert = decomp_ops.make<ov::op::v0::Convert>(downconvert, input_type);

            result = decomp_ops.make<ov::op::v1::Divide>(upconvert, input_scale);
        } else {
            const Output<Node> input_shift{fake_convert_node->input_value(2)};
            const auto shift = decomp_ops.make<ov::op::v1::Subtract>(scale, input_shift);

            const auto clamp = decomp_ops.make<ov::op::v0::Clamp>(shift, lower_bound, upper_bound);
            const auto downconvert =
                decomp_ops.make<ov::op::v0::Convert>(clamp, fake_convert_node->get_destination_element_type());
            const auto upconvert = decomp_ops.make<ov::op::v0::Convert>(downconvert, input_type);

            const auto deshift = decomp_ops.make<ov::op::v1::Add>(upconvert, input_shift);
            result = decomp_ops.make<ov::op::v1::Divide>(deshift, input_scale);
        }

        if (result->get_output_element_type(0) != fake_convert_node->get_output_element_type(0)) {
            result = decomp_ops.make<ov::op::v0::Convert>(result, fake_convert_node->get_output_element_type(0));
        }

        result->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(fake_convert_node, decomp_ops.get());
        ov::replace_node(m.get_match_root(), result);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fake_convert, matcher_name);
    register_matcher(m, callback);
}
