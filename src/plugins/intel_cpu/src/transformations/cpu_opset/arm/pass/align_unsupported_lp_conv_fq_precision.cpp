// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "align_unsupported_lp_conv_fq_precision.hpp"

#include <memory>

#include "conv_mul_add_fq_block.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/matcher.hpp"

using namespace ov::pass;

namespace {

bool is_int8_precision(const ov::element::Type& precision) {
    return (precision == ov::element::u8) || (precision == ov::element::i8);
}

}  // namespace

ov::intel_cpu::AlignUnsupportedLPConvFQPrecision::AlignUnsupportedLPConvFQPrecision() {
    auto conv_mul_add_fq = std::make_shared<ov::intel_cpu::ConvMulAddFQBlock>(true);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv_out = conv_mul_add_fq->get_anchor("convolution", pattern_map);
        const auto fq_out = conv_mul_add_fq->get_anchor("fake_quantize", pattern_map);
        if (!conv_out || !fq_out) {
            return false;
        }

        const auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(conv_out->get_node_shared_ptr());
        const auto fake_quantize = ov::as_type_ptr<ov::op::v0::FakeQuantize>(fq_out->get_node_shared_ptr());
        if (!conv || !fake_quantize) {
            return false;
        }

        if (fake_quantize->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        const auto activation_precision = conv->get_input_element_type(0);
        const auto output_precision = fake_quantize->get_output_element_type(0);
        if (!is_int8_precision(activation_precision) || !is_int8_precision(output_precision) ||
            (output_precision == activation_precision)) {
            return false;
        }

        ov::RTMap output_rt_info = fake_quantize->output(0).get_rt_info();
        auto precisions_attribute = ov::PrecisionsAttribute({activation_precision});
        const auto precisions_it = output_rt_info.find(ov::PrecisionsAttribute::get_type_info_static());
        if (precisions_it != output_rt_info.end()) {
            precisions_attribute = precisions_it->second.as<ov::PrecisionsAttribute>();
            precisions_attribute.value() = {activation_precision};
        }
        output_rt_info[ov::PrecisionsAttribute::get_type_info_static()] = precisions_attribute;

        const auto aligned_fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(
            ov::pass::low_precision::NetworkHelper::setOutDataPrecision(fake_quantize, activation_precision));
        if (!aligned_fq) {
            return false;
        }

        aligned_fq->set_friendly_name(fake_quantize->get_friendly_name());
        aligned_fq->output(0).get_rt_info() = output_rt_info;
        register_new_node(aligned_fq);
        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(conv_mul_add_fq, "AlignUnsupportedLPConvFQPrecision");
    register_matcher(matcher, callback);
}