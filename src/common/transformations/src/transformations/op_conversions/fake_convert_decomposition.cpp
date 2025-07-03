// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fake_convert_decomposition.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
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
    auto fake_convert_m = ov::pass::pattern::wrap_type<ov::op::v13::FakeConvert>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        const auto fake_convert =
            ov::as_type_ptr<ov::op::v13::FakeConvert>(pattern_to_output.at(fake_convert_m).get_node_shared_ptr());

        if (fake_convert == nullptr || transformation_callback(fake_convert)) {
            return false;
        }

        Output<Node> data{fake_convert->input_value(0)};
        const auto input_scale = fake_convert->input_value(1);
        auto input_type = data.get_element_type();

        ov::pass::NodeRegistry decomp_ops;
        if (input_type != input_scale.get_element_type()) {
            input_type = input_scale.get_element_type();
            data = decomp_ops.make<ov::op::v0::Convert>(data, input_type);
        }

        const auto shift = [&]() {
            if (fake_convert->get_input_size() == 2) {
                return ov::Output<ov::Node>{};
            }
            OPENVINO_ASSERT(fake_convert->get_input_size() == 3,
                            "FakeConvert should have either 2 or 3 inputs, but got: ",
                            fake_convert->get_input_size());
            return fake_convert->input_value(2);
        }();

        std::shared_ptr<Node> result = decomp_ops.make<ov::op::v1::Multiply>(data, input_scale);
        if (shift.get_node() != nullptr) {
            const auto& input_shift = fake_convert->input_value(2);
            result = decomp_ops.make<ov::op::v1::Subtract>(result, input_shift);
        }

        // Align with clamp behavior of FakeConvert in ngraph reference
        const auto [lower_bound, upper_bound] = [&]() {
            switch (fake_convert->get_destination_element_type()) {
                case ov::element::f8e4m3:
                    return std::make_pair(static_cast<double>(std::numeric_limits<ov::float8_e4m3>::lowest()),
                                          static_cast<double>(std::numeric_limits<ov::float8_e4m3>::max()));
                case ov::element::f8e5m2:
                    return std::make_pair(static_cast<double>(std::numeric_limits<ov::float8_e5m2>::lowest()),
                                          static_cast<double>(std::numeric_limits<ov::float8_e5m2>::max()));
                default:
                    OPENVINO_THROW("Unsupported destination element type: ", fake_convert->get_destination_element_type());
            }
        }();
        result = decomp_ops.make<ov::op::v0::Clamp>(result, lower_bound, upper_bound);
        result = decomp_ops.make<ov::op::v0::Convert>(result, fake_convert->get_destination_element_type());
        result = decomp_ops.make<ov::op::v0::Convert>(result, input_type);
        if (shift.get_node() != nullptr) {
            result = decomp_ops.make<ov::op::v1::Add>(result, shift);
        }
        result = decomp_ops.make<ov::op::v1::Divide>(result, input_scale);

        if (result->get_output_element_type(0) != fake_convert->get_output_element_type(0)) {
            result = decomp_ops.make<ov::op::v0::Convert>(result, fake_convert->get_output_element_type(0));
        }

        result->set_friendly_name(fake_convert->get_friendly_name());
        ov::copy_runtime_info(fake_convert, decomp_ops.get());
        ov::replace_node(m.get_match_root(), result);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fake_convert_m, matcher_name);
    register_matcher(m, callback);
}
