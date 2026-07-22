// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "align_unsupported_lp_conv_fq_precision.hpp"

#include <algorithm>
#include <memory>

#include "low_precision/network_helper.hpp"
#include "low_precision/resolve_precision_attribute.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/core/except.hpp"
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
    auto fq = wrap_type<ov::op::v0::FakeQuantize>({add, any_input(), any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv_node = pattern_map.at(conv).get_node_shared_ptr();
        auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(pattern_map.at(fq).get_node_shared_ptr());
        if (!fq_node) {
            return false;
        }

        auto conv_attr = low_precision::getAttribute<ov::PrecisionsAttribute>(conv_node->input(0));
        if (conv_attr.empty()) {
            return false;
        }
        const auto& conv_precisions = conv_attr.as<ov::PrecisionsAttribute>().value();
        if (conv_precisions.empty()) {
            return false;
        }
        OPENVINO_ASSERT(conv_precisions.size() == 1, "Convolution input precision should be single");
        const auto conv_precision = conv_precisions[0];

        auto fq_attr = low_precision::getAttributeFromOutput<ov::PrecisionsAttribute>(fq_node->output(0));
        if (fq_attr.empty()) {
            return false;
        }

        const auto& fq_precisions = fq_attr.as<ov::PrecisionsAttribute>().value();
        if (std::find(fq_precisions.begin(), fq_precisions.end(), conv_precision) == fq_precisions.end()) {
            return false;
        }

        ov::pass::low_precision::ResolvePrecisionAttribute::filterPrecisionsAttribute(fq_node);
        const auto fq_data_precision = ov::pass::low_precision::ResolvePrecisionAttribute::getDataPrecision(fq_node);
        if (fq_data_precision.empty()) {
            return false;
        }

        if (conv_precision == fq_data_precision.precision) {
            return false;
        }

        fq_attr.as<ov::PrecisionsAttribute>().value() = {conv_precision};

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(fq, "AlignUnsupportedLPConvFQPrecision");
    register_matcher(matcher, callback);
}