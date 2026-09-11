// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_conv_activation_zero_point.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"

ov::intel_cpu::ExtractConvActivationZeroPoint::ExtractConvActivationZeroPoint() {
    auto activation = ov::pass::pattern::any_input();
    auto zero_point = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto subtract = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({activation, zero_point});
    auto weights = ov::pass::pattern::any_input();
    auto conv =
        ov::pass::pattern::wrap_type<ov::op::v1::Convolution, ov::op::v1::GroupConvolution>({subtract, weights});

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto conv_node = m.get_match_root();
        if (!conv_node) {
            return false;
        }

        auto subtract_node = ov::as_type_ptr<ov::op::v1::Subtract>(conv_node->get_input_node_shared_ptr(0));
        if (!subtract_node) {
            return false;
        }

        auto zero_point_node = ov::as_type_ptr<ov::op::v0::Constant>(subtract_node->get_input_node_shared_ptr(1));
        if (!zero_point_node) {
            return false;
        }

        const auto zp = zero_point_node->cast_vector<float>();
        if (zp.empty()) {
            return false;
        }
        const auto zp_value = zp[0];
        if (zp_value != std::round(zp_value) || !std::all_of(zp.begin(), zp.end(), [zp_value](float value) {
                return value == zp_value;
            })) {
            return false;
        }
        const auto offset = static_cast<int32_t>(zp_value);

        const auto activation = subtract_node->input_value(0);
        const auto activation_type = activation.get_element_type();

        bool tail_has_swish = false;
        auto tail_type = ov::element::dynamic;
        auto tail_output = conv_node->output(0);
        for (int depth = 0; depth < 5; ++depth) {
            const auto& consumers = tail_output.get_target_inputs();
            if (consumers.size() != 1) {
                break;
            }
            auto* consumer = consumers.begin()->get_node();
            if (ov::is_type<ov::op::v0::FakeQuantize>(consumer)) {
                tail_type = consumer->get_output_element_type(0);
                break;
            }
            if (ov::is_type<ov::op::v4::Swish>(consumer)) {
                tail_has_swish = true;
            } else if (!ov::is_type<ov::op::v1::Multiply>(consumer) && !ov::is_type<ov::op::v1::Add>(consumer)) {
                break;
            }
            tail_output = consumer->output(0);
        }

        if (activation_type == ov::element::u8 && tail_type == ov::element::u8) {
            if (tail_has_swish) {
                return false;
            }
            conv_node->get_rt_info()[rt_info_key] = static_cast<int32_t>(offset);
            conv_node->input(0).replace_source_output(activation);
        } else if (activation_type == ov::element::u8) {
            if (tail_type != ov::element::i8) {
                const auto& conv_consumers = conv_node->output(0).get_target_inputs();
                if (conv_consumers.size() != 1) {
                    return false;
                }
                auto* dequant_multiply = conv_consumers.begin()->get_node();
                if (!ov::is_type<ov::op::v1::Multiply>(dequant_multiply)) {
                    return false;
                }
                const auto scales =
                    ov::as_type_ptr<ov::op::v0::Constant>(dequant_multiply->input_value(1).get_node_shared_ptr());
                if (!scales || ov::shape_size(scales->get_shape()) != 1) {
                    return false;
                }
            }
            constexpr int32_t shift = 128;
            const auto fq_node = activation.get_node_shared_ptr();
            const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(fq_node);
            const auto fq_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(fq_node);
            if (!fq || !fq_relaxed || activation.get_target_inputs().size() != 1) {
                return false;
            }
            constexpr size_t out_low_port = 3;
            constexpr size_t out_high_port = 4;
            for (size_t port : {out_low_port, out_high_port}) {
                if (!ov::as_type_ptr<ov::op::v0::Constant>(fq_node->get_input_node_shared_ptr(port))) {
                    return false;
                }
            }

            for (size_t port : {out_low_port, out_high_port}) {
                const auto bound = ov::as_type_ptr<ov::op::v0::Constant>(fq_node->get_input_node_shared_ptr(port));
                auto values = bound->cast_vector<float>();
                for (auto& value : values) {
                    value -= static_cast<float>(shift);
                }
                const auto shifted =
                    ov::op::v0::Constant::create(bound->get_element_type(), bound->get_shape(), values);
                fq_node->input(port).replace_source_output(shifted->output(0));
            }
            fq_relaxed->set_overridden_output_type(ov::element::i8, 0);
            fq_node->validate_and_infer_types();

            conv_node->get_rt_info()[rt_info_key] = static_cast<int32_t>(offset - shift);
            conv_node->input(0).replace_source_output(activation);
        } else if (activation_type == ov::element::i8) {
            conv_node->get_rt_info()[rt_info_key] = static_cast<int32_t>(offset);
            conv_node->input(0).replace_source_output(activation);
        } else {
            return false;
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv, "ExtractConvActivationZeroPoint");
    register_matcher(m, callback);
}
