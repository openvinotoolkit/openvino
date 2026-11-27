// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_power_half_to_sqrt.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

bool is_standard_float(const ov::element::Type& et) {
    return et.is_real() && !et.is_quantized();
}

}  // namespace

ConvertPowerHalfToSqrt::ConvertPowerHalfToSqrt() {
    using namespace ov::pass::pattern;

    auto x = any_input([](const ov::Output<ov::Node>& output) {
        return is_standard_float(output.get_element_type());
    });
    auto const_half = wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        const auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
        if (!const_node || !is_standard_float(const_node->get_element_type()) ||
            !ov::op::util::is_scalar_or_single_elem_constant(const_node)) {
            return false;
        }
        return const_node->cast_vector<double>()[0] == 0.5;
    });
    auto opt_convert = optional<ov::op::v0::Convert>({const_half}, [](const ov::Output<ov::Node>& output) {
        return is_standard_float(output.get_element_type());
    });
    auto power = wrap_type<ov::op::v1::Power>({x, opt_convert}, [](const ov::Output<ov::Node>& output) {
        const auto* node = output.get_node();
        if (output.get_element_type() != node->get_input_element_type(0)) {
            return false;
        }
        const auto& in0_shape = node->get_input_partial_shape(0);
        const auto& in1_shape = node->get_input_partial_shape(1);
        const auto& out_shape = output.get_partial_shape();
        if (in1_shape.rank().is_dynamic()) {
            return false;
        }
        return in1_shape.rank().get_length() == 0 ||
               (in0_shape.rank().is_static() && out_shape.rank().is_static() && in0_shape == out_shape);
    });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto power_node = m.get_match_root();
        if (transformation_callback(power_node)) {
            return false;
        }

        auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(power_node->input_value(0));
        sqrt_node->set_friendly_name(power_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sqrt_node);
        ov::replace_node(power_node, sqrt_node);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(power, "ConvertPowerHalfToSqrt"), callback);
}

}  // namespace ov::intel_gpu
