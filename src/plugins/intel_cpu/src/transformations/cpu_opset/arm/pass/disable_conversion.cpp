// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "disable_conversion.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

#include <openvino/opsets/opset6.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_cpu::DisableConversion::DisableConversion() {
    auto mvn = ngraph::pattern::wrap_type<ov::op::v6::MVN>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!node)
            return false;

        disable_fp16_compression(node);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(mvn, "DisableConversion");
    register_matcher(m, callback);
}
