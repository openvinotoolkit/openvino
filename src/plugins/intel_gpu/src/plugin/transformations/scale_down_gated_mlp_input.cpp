// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale_down_gated_mlp_input.hpp"

#include <memory>

#include "intel_gpu/op/fully_connected.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ScaleDownGatedMLPInput::ScaleDownGatedMLPInput(float scale_factor) {
    using namespace ov::pass::pattern;

    // FullyConnected and FullyConnectedCompressed take a varying number of inputs, so the
    // gated product on input 0 is checked in the callback.
    auto down_m = wrap_type<op::FullyConnected>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto down = m.get_match_root();
        if (transformation_callback(down))
            return false;

        const auto product = ov::as_type_ptr<ov::op::v1::Multiply>(down->get_input_node_shared_ptr(0));
        if (!product || product->get_output_element_type(0) != ov::element::f16 || product->get_output_target_inputs(0).size() != 1)
            return false;
        std::shared_ptr<ov::Node> gate;
        for (const auto& in : product->input_values()) {
            if (ov::is_type<ov::op::v4::Swish>(in.get_node()) && in.get_target_inputs().size() == 1)
                gate = in.get_node_shared_ptr();
        }
        if (!gate)
            return false;

        const auto gate_consumers = gate->get_output_target_inputs(0);
        auto scale_down =
            std::make_shared<ov::op::v1::Multiply>(gate, ov::op::v0::Constant::create(gate->get_output_element_type(0), ov::Shape{}, {1.f / scale_factor}));
        scale_down->set_friendly_name(down->get_friendly_name() + "_scale_down");
        ov::copy_runtime_info(down, scale_down);
        for (const auto& in : gate_consumers)
            in.replace_source_output(scale_down);

        // The scale-up goes after the Convert that keeps the projection output in FP32, if there is one.
        std::shared_ptr<ov::Node> out = down;
        const auto down_consumers = down->get_output_target_inputs(0);
        if (down_consumers.size() == 1) {
            auto convert = down_consumers.begin()->get_node()->shared_from_this();
            if (ov::is_type<ov::op::v0::Convert>(convert) && ov::is_conversion_disabled(convert, ov::element::f16))
                out = convert;
        }
        const auto out_consumers = out->get_output_target_inputs(0);
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(out, ov::op::v0::Constant::create(out->get_output_element_type(0), ov::Shape{}, {scale_factor}));
        scale_up->set_friendly_name(out->get_friendly_name() + "_scale_up");
        ov::copy_runtime_info(down, scale_up);
        scale_up->output(0).set_names(out->output(0).get_names());
        out->output(0).set_names({});
        for (const auto& in : out_consumers)
            in.replace_source_output(scale_up);
        return true;
    };

    auto m = std::make_shared<Matcher>(down_m, "ScaleDownGatedMLPInput");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
