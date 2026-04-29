// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "align_unsupported_lp_conv_fq_precision.hpp"

#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass;

ov::intel_cpu::AlignUnsupportedLPConvFQPrecision::AlignUnsupportedLPConvFQPrecision() {
    using namespace ov::pass::pattern;

    auto conv = wrap_type<ov::op::v1::Convolution>({any_input(), any_input()});
    auto add = wrap_type<ov::op::v1::Add>({conv, any_input()});
    auto fq = wrap_type<ov::op::v0::FakeQuantize>(
        {add, any_input(), any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv_node = pattern_map.at(conv).get_node_shared_ptr();
        auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(
            pattern_map.at(fq).get_node_shared_ptr());
        if (!fq_node) {
            return false;
        }

        auto input_fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(
            conv_node->get_input_node_shared_ptr(0));
        if (!input_fq) {
            return false;
        }

        const auto input_data_precision =
            low_precision::fq_decomposition::getDataPrecisionByOutputPort(input_fq);
        if (input_data_precision.empty()) {
            return false;
        }

        auto conv_attr = low_precision::getAttribute<ov::PrecisionsAttribute>(conv_node->input(0));
        if (conv_attr.empty()) {
            return false;
        }
        const auto& conv_precisions = conv_attr.as<ov::PrecisionsAttribute>().value();
        if (conv_precisions.empty() || conv_precisions.size() != 1) {
            return false;
        }
        const auto conv_precision = conv_precisions[0];

        const auto fq_data_precision =
            low_precision::fq_decomposition::getDataPrecisionByOutputPort(fq_node);
        if (fq_data_precision.empty()) {
            return false;
        }

        if (conv_precision == fq_data_precision.precision) {
            return false;
        }

        auto fq_attr = low_precision::getAttributeFromOutput<ov::PrecisionsAttribute>(
            fq_node->output(0));
        if (fq_attr.empty()) {
            fq_node->output(0).get_rt_info()[ov::PrecisionsAttribute::get_type_info_static()] =
                ov::PrecisionsAttribute({conv_precision});
        } else {
            fq_attr.as<ov::PrecisionsAttribute>().value() = {conv_precision};
        }

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(fq, "AlignUnsupportedLPConvFQPrecision");
    register_matcher(matcher, callback);
}