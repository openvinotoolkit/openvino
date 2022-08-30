// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_segments_creator_identity.hpp"

#include "gna_plugin_log.hpp"
#include "gna_slope_scale.h"
#include "pwl_input_params.hpp"
#include "pwl_tools.hpp"
#include "round_float_define.hpp"
#include "runtime/pwl.h"

namespace ov {
namespace intel_gna {
namespace backend {

PWLSegmentsCreatorIdentity::PWLSegmentsCreatorIdentity(std::shared_ptr<PWLBorderValuesCounter> border_counter)
    : border_counter_(std::move(border_counter)) {
    if (border_counter_ == nullptr) {
        THROW_GNA_EXCEPTION << "Passed border_counter() is nullptr";
    }
}

std::vector<gna_pwl_segment_t> PWLSegmentsCreatorIdentity::CreateSegments(const PWLInputParams& input_params) const {
    const auto border_valus = border_counter_->CreateBorderValues(input_params);
    return CreateSegments(input_params, border_valus);
}

PWLSegmentsWithBorderValues PWLSegmentsCreatorIdentity::CreateSegmentsWithBorders(
    const PWLInputParams& input_params) const {
    const auto border_valus = border_counter_->CreateBorderValues(input_params);
    auto segments = CreateSegments(input_params, border_valus);
    return {border_valus, segments};
}

std::vector<gna_pwl_segment_t> PWLSegmentsCreatorIdentity::CreateSegments(const PWLInputParams& input_params,
                                                                          const BorderValues& border_values) const {
    std::vector<gna_pwl_segment_t> segments;

    // first segment
    int32_t x_base = static_cast<int32_t>(INT32_MIN & XBASEMASK);
    segments.push_back({x_base, border_values.y_lower, 0});

    // first segment
    auto gna_slope_segment_1 = gna_slope(1.0, input_params.in_scale(), input_params.out_scale());
    x_base = static_cast<int32_t>(border_values.x_lower & XBASEMASK);
    x_base |= gna_slope_segment_1.slope_scale_index;
    auto slope_segment_1 = FLOAT_TO_INT16(gna_slope_segment_1.slope * gna_slope_segment_1.slope_scale);
    segments.push_back({x_base, border_values.y_lower, slope_segment_1});

    // It was decided based on 89164 to introduce extra segment to ensure that PWL for Identity will cross (0,0).
    // count zero of function for middle segment.
    auto y0 = PWLTools::ComputePWL(segments[1], 0);
    // TODO remove when fixed
    std::cout << "y0: " << y0 << std::endl;
    if (y0 > border_values.y_upper || y0 < border_values.y_lower) {
        // TODO remove when fixed
        std::cout << "Invalid parameters. F(0)=" << y0 << " exceedes allowed values <" << border_values.y_lower << ", "
                  << border_values.y_upper << ">";
        THROW_GNA_EXCEPTION << "Invalid parameters. F(0)=" << y0 << " exceedes allowed values <"
                            << border_values.y_lower << ", " << border_values.y_upper << ">";
    }

    // if y0 == 0 we just add third segment if needed and return
    if (y0 == 0) {
        if (INT32_MAX > border_values.x_upper) {
            x_base = static_cast<int32_t>(border_values.x_upper & XBASEMASK);
            segments.push_back({x_base, border_values.y_upper, 0});
        }
        return segments;
    }

    gnalog() << "PWL does not pass (0,0), F(0)=" << y0 << "! Adjusting PWL segments.";

    // y0 != 0 add new segment, update previous one and cound properly next one if needed.

    // create a new segment with xBase = 0 and yBase = 0
    // use the same slope and scale as for previous segment
    x_base = static_cast<int32_t>(0 | gna_slope_segment_1.slope_scale_index);
    x_base |= gna_slope_segment_1.slope_scale_index;
    segments.push_back({x_base, 0, slope_segment_1});
    auto& extra_segment = segments.back();

    // adapt xBase for point on the left side of 0,0 to ensure that F(-1) <= F(0)
    UpdateSegmentOnTheLeftOf0_0(segments[0], segments[1], y0, gna_slope_segment_1.slope_scale_index);

    if (INT32_MAX > border_values.x_upper) {
        auto last_segment = CalculcateLastSegmentOnTheRightOf0_0(extra_segment, border_values);
        segments.push_back(last_segment);
    }

    return segments;
}

void PWLSegmentsCreatorIdentity::UpdateSegmentOnTheLeftOf0_0(const gna_pwl_segment_t& segment_0,
                                                             gna_pwl_segment_t& segment_1,
                                                             const int64_t delta_y,
                                                             uint32_t slope_scale_index) const {
    // TODO remove when agreed
    // new_segment_1_x_base = segment_1.xBase - (segment_0.yBase - segment_1.yBase - y_delta) * segment_1.scale_factor /
    // segment_1.slope
    auto segment_0_in_int64 = PWLTools::ConvertSegementTo64(segment_0);
    auto segment_1_in_int64 = PWLTools::ConvertSegementTo64(segment_1);

    if (segment_1_in_int64.slope == 0) {
        THROW_GNA_EXCEPTION << "Slope is 0 possible division by 0 when updating left segment!.";
    }
    int64_t new_segment_1_x_base =
        segment_1_in_int64.x_base - (segment_0_in_int64.y_base - segment_1_in_int64.y_base - delta_y) *
                                        segment_1_in_int64.slope_scale / segment_1_in_int64.slope;

    new_segment_1_x_base = PWLTools::RoundTowardZero(new_segment_1_x_base);
    const auto new_segment_1_x_base_32 = static_cast<int32_t>(new_segment_1_x_base);
    segment_1.xBase = static_cast<int32_t>(new_segment_1_x_base_32 & XBASEMASK);
    segment_1.xBase |= slope_scale_index;
}

gna_pwl_segment_t PWLSegmentsCreatorIdentity::CalculcateLastSegmentOnTheRightOf0_0(
    const gna_pwl_segment_t& segment_0_0,
    const BorderValues& border_values) const {
    // TODO remove when agreed
    // right_segment_x_base = segment_0_0.xBase + (border_values.y_upper - segment_0_0.yBase ) *
    // segment_0_0.scale_facotr / segment_0_0.slope
    auto segment_0_0_in_int64 = PWLTools::ConvertSegementTo64(segment_0_0);
    int64_t right_y_base_64 = static_cast<int64_t>(border_values.y_upper);

    if (segment_0_0_in_int64.slope == 0) {
        THROW_GNA_EXCEPTION << "Slope is 0 possible division by 0 when calculating right segment!.";
    }
    // maybe calculation should be done no float/doubles and at the end converted to int32_t
    int64_t right_segment_x_base = segment_0_0_in_int64.x_base + (right_y_base_64 - segment_0_0_in_int64.y_base) *
                                                                     segment_0_0_in_int64.slope_scale /
                                                                     segment_0_0_in_int64.slope;

    right_segment_x_base = PWLTools::RoundTowardZero(right_segment_x_base);
    int32_t right_segment_x_base_32 = static_cast<int32_t>(right_segment_x_base);
    right_segment_x_base_32 = static_cast<int32_t>(right_segment_x_base_32 & XBASEMASK);  // reset 2
    // don't set slope_index due the fact slope is 0
    return {right_segment_x_base_32, border_values.y_upper, 0};
}

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov