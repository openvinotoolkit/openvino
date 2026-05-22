// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_clamp_and_fake_quantize.hpp"

#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

bool clamp_covers_fake_quantize_interval(const std::shared_ptr<ov::op::v0::Clamp>& clamp,
                                         const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
    const auto input_low = ov::as_type_ptr<const ov::op::v0::Constant>(fq->get_input_node_shared_ptr(1));
    const auto input_high = ov::as_type_ptr<const ov::op::v0::Constant>(fq->get_input_node_shared_ptr(2));
    if (!input_low || !input_high) {
        return false;
    }

    const auto input_low_values = input_low->cast_vector<float>();
    const auto input_high_values = input_high->cast_vector<float>();
    const auto clamp_low = static_cast<float>(clamp->get_min());
    const auto clamp_high = static_cast<float>(clamp->get_max());

    for (const auto value : input_low_values) {
        if (clamp_low > value) {
            return false;
        }
    }

    for (const auto value : input_high_values) {
        if (clamp_high < value) {
            return false;
        }
    }

    return true;
}

}  // namespace

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
        if (!fq) {
            return false;
        }

        auto current_output = fq->input_value(0);
        auto clamp = ov::as_type_ptr<ov::op::v0::Clamp>(current_output.get_node_shared_ptr());
        if (!clamp) {
            return false;
        }

        auto removed_redundant_clamp = false;
        while ((clamp = ov::as_type_ptr<ov::op::v0::Clamp>(current_output.get_node_shared_ptr()))) {
            if (!clamp_covers_fake_quantize_interval(clamp, fq)) {
                break;
            }

            current_output = clamp->input_value(0);
            removed_redundant_clamp = true;
        }

        if (!removed_redundant_clamp) {
            return false;
        }

        fq->input(0).replace_source_output(current_output);
        fq->validate_and_infer_types();
        return true;
    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(fake_quantize, matcher_name);
    register_matcher(matcher, callback);
}