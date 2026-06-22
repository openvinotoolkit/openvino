// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_eliminate_sequential.hpp"

#include <cmath>
#include <map>
#include <memory>
#include <string>

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
    auto p_fq1 = pattern::wrap_type<v0::FakeQuantize>(
        {pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        pattern::consumers_count(1));
    auto p_fq2 = pattern::wrap_type<v0::FakeQuantize>(
        {p_fq1, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});

    // Folds two sequential FakeQuantize ops (FQ1 -> FQ2) into one. Notation below:
    // FQ(in_low, in_high, out_low, out_high, levels).
    //   - Elimination (FQ2 dropped, FQ1 kept), e.g.
    //       FQ1(-1, 1, -1, 1, 256) -> FQ2(-2, 2, -2, 2, 1021)  =>  FQ1(-1, 1, -1, 1, 256)
    //   - Merge into a single FQ (FQ1 input range -> FQ2 output range), e.g.
    //       FQ1(-2, 2, -1, 1, 256) -> FQ2(-1, 1, -1, 0, 256)    =>  FQ(-2, 2, -1, 0, 256)
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto fq1 = ov::as_type_ptr<v0::FakeQuantize>(pattern_map.at(p_fq1).get_node_shared_ptr());
        auto fq2 = ov::as_type_ptr<v0::FakeQuantize>(pattern_map.at(p_fq2).get_node_shared_ptr());

        if (!fq1 || !fq2) {
            return false;
        }

        std::map<std::string, double> fq1_ranges = {{"in.low", 0.0},
                                                    {"in.high", 0.0},
                                                    {"out.low", 0.0},
                                                    {"out.high", 0.0}};
        std::map<std::string, double> fq2_ranges = {{"in.low", 0.0},
                                                    {"in.high", 0.0},
                                                    {"out.low", 0.0},
                                                    {"out.high", 0.0}};

        // Extract all four range bounds of a FakeQuantize and verify they are finite constants.
        auto get_ranges = [](const std::shared_ptr<v0::FakeQuantize>& fq,
                             std::map<std::string, double>& ranges) -> bool {
            if (!ov::op::util::get_constant_value(fq->input_value(1).get_node_shared_ptr(), ranges["in.low"]) ||
                !ov::op::util::get_constant_value(fq->input_value(2).get_node_shared_ptr(), ranges["in.high"]) ||
                !ov::op::util::get_constant_value(fq->input_value(3).get_node_shared_ptr(), ranges["out.low"]) ||
                !ov::op::util::get_constant_value(fq->input_value(4).get_node_shared_ptr(), ranges["out.high"])) {
                return false;
            }
            return std::isfinite(ranges["in.low"]) && std::isfinite(ranges["in.high"]) &&
                   std::isfinite(ranges["out.low"]) && std::isfinite(ranges["out.high"]);
        };

        if (!get_ranges(fq1, fq1_ranges) || !get_ranges(fq2, fq2_ranges)) {
            return false;
        }

        // FQ1 output must stay within FQ2 input range, otherwise FQ2 clamps it and the
        // composition is not equivalent to a single FakeQuantize.
        // Rejected: FQ1(-1, 1, -1, 1, 256) -> FQ2(-0.5, 0.5, -1, 1, 256), since FQ1 emits values
        // in [-1, 1] that exceed FQ2 input range [-0.5, 0.5] and get clamped.
        if (fq1_ranges["out.low"] < fq2_ranges["in.low"] || fq1_ranges["out.high"] > fq2_ranges["in.high"]) {
            return false;
        }

        // Both grids must be non-degenerate to define a quantization step.
        const auto fq1_levels = static_cast<double>(fq1->get_levels());
        const auto fq2_levels = static_cast<double>(fq2->get_levels());
        if (fq1_levels <= 1.0 || fq2_levels <= 1.0) {
            return false;
        }

        const double fq1_step = (fq1_ranges["out.high"] - fq1_ranges["out.low"]) / (fq1_levels - 1.0);
        const double fq2_step = (fq2_ranges["out.high"] - fq2_ranges["out.low"]) / (fq2_levels - 1.0);

        // FQ1 output grid must align with FQ2 output grid so re-quantization preserves every FQ1 level.
        // Aligned (FQ1 step is a multiple of FQ2 step, on-grid):
        //   FQ1(-1, 1, -1, 1, 256) -> FQ2(-2, 2, -2, 2, 1021): fq1_step = 2/255, fq2_step = 4/1020 = 1/255.
        // Rejected (off-grid): FQ1(-1, 1, -1, 1, 256) -> FQ2(-2, 2, -2, 2, 256), fq1_step = 2/255 is not
        // an integer multiple of fq2_step = 4/255.
        if (!is_integer_multiple(fq1_ranges["out.low"] - fq2_ranges["out.low"], fq2_step) ||
            !is_integer_multiple(fq1_step, fq2_step)) {
            return false;
        }

        // FQ2 maps its input range onto itself (in == out): it only re-quantizes onto an aligned
        // super-grid and therefore is redundant once the grids align -> drop FQ2, keep FQ1.
        // Eliminated: FQ1(-1, 1, -1, 1, 256) -> FQ2(-2, 2, -2, 2, 1021)  =>  FQ1(-1, 1, -1, 1, 256).
        const bool fq2_is_identity = is_close(fq2_ranges["in.low"], fq2_ranges["out.low"]) &&
                                     is_close(fq2_ranges["in.high"], fq2_ranges["out.high"]);
        if (fq2_is_identity) {
            return replace_output_update_name(fq2->output(0), fq1->output(0));
        }

        // Otherwise merge FQ1 and FQ2 into a single FakeQuantize: FQ1 input range -> FQ2 output range.
        // Equal levels are required so the merged grid reproduces both quantization steps.
        // Merged: FQ1(-2, 2, -1, 1, 256) -> FQ2(-1, 1, -1, 0, 256)  =>  FQ(-2, 2, -1, 0, 256).
        if (fq1->get_levels() != fq2->get_levels()) {
            return false;
        }

        // Create merged FQ with FQ1's input range and FQ2's output range.
        auto fq1_in_low_const =
            v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq1_ranges["in.low"])});
        auto fq1_in_high_const =
            v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq1_ranges["in.high"])});
        auto fq2_out_low_const =
            v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq2_ranges["out.low"])});
        auto fq2_out_high_const =
            v0::Constant::create(element::f32, Shape{1}, {static_cast<float>(fq2_ranges["out.high"])});

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

    auto m = std::make_shared<pattern::Matcher>(p_fq2, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
