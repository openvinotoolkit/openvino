// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_clamp_and_fake_quantize.hpp"

#include <memory>
#include <utility>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils/rt_info/fake_quantize_clamp_bounds.hpp"

ov::intel_cpu::FuseClampAndFakeQuantize::FuseClampAndFakeQuantize() {
    MATCHER_SCOPE(FuseClampAndFakeQuantize);

    auto input = ov::pass::pattern::any_input();
    auto clamp = ov::pass::pattern::wrap_type<ov::op::v0::Clamp>({input});
    auto fake_quantize = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {clamp,
         ov::pass::pattern::any_input(),
         ov::pass::pattern::any_input(),
         ov::pass::pattern::any_input(),
         ov::pass::pattern::any_input()});

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(m.get_match_root());
        if (!fq || fq->get_levels() == 2) {
            return false;
        }

        auto current_output = fq->input_value(0);
        auto clamp = ov::as_type_ptr<ov::op::v0::Clamp>(current_output.get_node_shared_ptr());
        if (!clamp) {
            return false;
        }

        auto fused_low = 0.f;
        auto fused_high = 0.f;
        auto has_bounds = false;
        if (const auto existing_bounds = ov::intel_cpu::get_fake_quantize_clamp_bounds(fq)) {
            fused_low = existing_bounds->low();
            fused_high = existing_bounds->high();
            has_bounds = true;
        }

        while ((clamp = ov::as_type_ptr<ov::op::v0::Clamp>(current_output.get_node_shared_ptr()))) {
            const auto clamp_low = static_cast<float>(clamp->get_min());
            const auto clamp_high = static_cast<float>(clamp->get_max());

            if (!has_bounds) {
                fused_low = clamp_low;
                fused_high = clamp_high;
                has_bounds = true;
            } else {
                std::tie(fused_low, fused_high) =
                    ov::intel_cpu::compose_clamp_intervals(clamp_low, clamp_high, fused_low, fused_high);
            }

            current_output = clamp->input_value(0);
        }

        ov::intel_cpu::set_fake_quantize_clamp_bounds(fq, fused_low, fused_high);
        fq->input(0).replace_source_output(current_output);
        fq->validate_and_infer_types();
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(fake_quantize, matcher_name);
    register_matcher(matcher, callback);
}