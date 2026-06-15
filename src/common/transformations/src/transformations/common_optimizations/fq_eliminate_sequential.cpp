// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_eliminate_sequential.hpp"

#include <cmath>
#include <memory>

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;

namespace ov::pass {

namespace {

bool inputs_equal_or_same_constant(const Output<Node>& lhs, const Output<Node>& rhs) {
    if (lhs == rhs) {
        return true;
    }

    auto lhs_const = ov::as_type_ptr<v0::Constant>(lhs.get_node_shared_ptr());
    auto rhs_const = ov::as_type_ptr<v0::Constant>(rhs.get_node_shared_ptr());
    if (!lhs_const || !rhs_const) {
        return false;
    }

    return ov::compare_constants(lhs_const, rhs_const);
}

bool is_close(double lhs, double rhs, double eps = 1e-12) {
    return std::fabs(lhs - rhs) <= eps * (1.0 + std::max(std::fabs(lhs), std::fabs(rhs)));
}

bool is_integer_multiple(double value, double step, double eps = 1e-9) {
    if (step == 0.0) {
        return is_close(value, 0.0, eps);
    }

    const double ratio = value / step;
    return is_close(ratio, std::round(ratio), eps);
}

}  // namespace

FakeQuantizeEliminateSequential::FakeQuantizeEliminateSequential() {
    MATCHER_SCOPE(FakeQuantizeEliminateSequential);
    auto fq1 = pattern::wrap_type<v0::FakeQuantize>(
        {pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});
    auto fq2 = pattern::wrap_type<v0::FakeQuantize>(
        {fq1, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq2 = ov::as_type_ptr<v0::FakeQuantize>(m.get_match_root());
        if (!fq2) {
            return false;
        }

        auto fq1 = ov::as_type_ptr<v0::FakeQuantize>(fq2->input_value(0).get_node_shared_ptr());
        if (!fq1) {
            return false;
        }

        bool all_inputs_equal = true;
        for (size_t i = 1; i < fq1->get_input_size(); ++i) {
            if (!inputs_equal_or_same_constant(fq1->input_value(i), fq2->input_value(i))) {
                all_inputs_equal = false;
                break;
            }
        }

        // Case 1: identical parameters and levels -> second FQ is redundant.
        if (all_inputs_equal && fq1->get_levels() == fq2->get_levels()) {
            return replace_output_update_name(fq2->output(0), fq1->output(0));
        }

        double fq1_out_low = 0.0;
        double fq1_out_high = 0.0;
        double fq1_in_low = 0.0;
        double fq1_in_high = 0.0;
        double fq2_in_low = 0.0;
        double fq2_in_high = 0.0;
        double fq2_out_low = 0.0;
        double fq2_out_high = 0.0;

        if (!ov::op::util::get_constant_value(fq1->input_value(1).get_node_shared_ptr(), fq1_in_low) ||
            !ov::op::util::get_constant_value(fq1->input_value(2).get_node_shared_ptr(), fq1_in_high) ||
            !std::isfinite(fq1_in_low) || !std::isfinite(fq1_in_high)) {
            return false;
        }
        if (fq1_in_high <= fq1_in_low) {
            return false;
        }

        if (!ov::op::util::get_constant_value(fq1->input_value(3).get_node_shared_ptr(), fq1_out_low) ||
            !ov::op::util::get_constant_value(fq1->input_value(4).get_node_shared_ptr(), fq1_out_high) ||
            !ov::op::util::get_constant_value(fq2->input_value(1).get_node_shared_ptr(), fq2_in_low) ||
            !ov::op::util::get_constant_value(fq2->input_value(2).get_node_shared_ptr(), fq2_in_high) ||
            !ov::op::util::get_constant_value(fq2->input_value(3).get_node_shared_ptr(), fq2_out_low) ||
            !ov::op::util::get_constant_value(fq2->input_value(4).get_node_shared_ptr(), fq2_out_high) ||
            !std::isfinite(fq1_out_low) || !std::isfinite(fq1_out_high) || !std::isfinite(fq2_in_low) ||
            !std::isfinite(fq2_in_high) || !std::isfinite(fq2_out_low) || !std::isfinite(fq2_out_high)) {
            return false;
        }

        // Check if FQ1 output range is within FQ2 input range (necessary for any optimization).
        if (fq1_out_low < fq2_in_low || fq1_out_high > fq2_in_high) {
            return false;
        }

        // Case 2: handle subrange/identity path only when FQ2 is identity on its own range.
        const bool fq2_is_identity = is_close(fq2_in_low, fq2_out_low) && is_close(fq2_in_high, fq2_out_high);

        // Case 3: FQ1 output must lie within the FQ2 output range (only for elimination case).
        if (fq2_is_identity && (fq1_out_low < fq2_out_low || fq1_out_high > fq2_out_high)) {
            return false;
        }

        const auto fq2_levels = static_cast<double>(fq2->get_levels());
        if (fq2_levels <= 1.0) {
            return false;
        }

        const double fq2_step = (fq2_out_high - fq2_out_low) / (fq2_levels - 1.0);
        // Case 4: degenerate FQ2 grid -> remove only if ranges match exactly.
        if (is_close(fq2_step, 0.0)) {
            if (is_close(fq1_out_low, fq2_out_low) && is_close(fq1_out_high, fq2_out_high)) {
                return replace_output_update_name(fq2->output(0), fq1->output(0));
            }
            return false;
        }

        const auto fq1_levels = static_cast<double>(fq1->get_levels());
        if (fq1_levels <= 1.0) {
            return false;
        }

        const double fq1_step = (fq1_out_high - fq1_out_low) / (fq1_levels - 1.0);
        // Case 5: FQ1 grid must align to FQ2 grid (may have different levels).
        if (!is_integer_multiple(fq1_out_low - fq2_out_low, fq2_step) || !is_integer_multiple(fq1_step, fq2_step)) {
            return false;
        }

        // If we reach here, grids align.
        if (fq2_is_identity) {
            // Case 1-5 elimination: FQ2 is identity and grids align -> eliminate FQ2.
            return replace_output_update_name(fq2->output(0), fq1->output(0));
        }

        // Case 6: Merge FQ1 and FQ2 into a single FQ when grids align but FQ2 is not identity.
        // Both must have the same levels to ensure mathematical correctness.
        if (fq1->get_levels() != fq2->get_levels()) {
            return false;
        }

        // Create merged FQ with FQ1's input range and FQ2's output range.
        auto fq1_in_low_const = v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq1_in_low)});
        auto fq1_in_high_const = v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq1_in_high)});
        auto fq2_out_low_const = v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq2_out_low)});
        auto fq2_out_high_const = v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq2_out_high)});

        auto merged_fq = std::make_shared<v0::FakeQuantize>(fq1->input_value(0),
                                                            fq1_in_low_const,
                                                            fq1_in_high_const,
                                                            fq2_out_low_const,
                                                            fq2_out_high_const,
                                                            fq1->get_levels());

        merged_fq->set_friendly_name(fq2->get_friendly_name());
        ov::copy_runtime_info({fq1, fq2}, merged_fq);
        ov::replace_node(fq2, merged_fq);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(fq2, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
