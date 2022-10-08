// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_segments_creator_identity.hpp"

#include "gna_plugin_log.hpp"
#include "gna_slope_scale.h"
#include "pwl_input_params.hpp"
#include "pwl_tools.hpp"
#include "runtime/pwl.h"

namespace ov {
namespace intel_gna {
namespace backend {
using namespace pwl_tools;

PWLSegmentsCreatorIdentity::PWLSegmentsCreatorIdentity(std::shared_ptr<PWLBorderValuesCounter> border_counter)
    : border_counter_(std::move(border_counter)) {
    if (border_counter_ == nullptr) {
        THROW_GNA_EXCEPTION << "Passed border_counter() is nullptr";
    }
}

std::vector<gna_pwl_segment_t> PWLSegmentsCreatorIdentity::CreateSegments(const PWLInputParams& input_params) const {
    const auto border_values = border_counter_->CreateBorderValues(input_params);
    return CreateSegments(input_params, border_values);
}

PWLSegmentsWithBorderValues PWLSegmentsCreatorIdentity::CreateSegmentsWithBorders(
    const PWLInputParams& input_params) const {
    const auto border_values = border_counter_->CreateBorderValues(input_params);
    auto segments = CreateSegments(input_params, border_values);
    return {border_values, segments};
}

std::vector<gna_pwl_segment_t> PWLSegmentsCreatorIdentity::CreateSegments(const PWLInputParams& input_params,
                                                                          const BorderValues& border_values) const {
    std::vector<gna_pwl_segment_t> segments;

    // segment 0
    segments.push_back(CreateSegment0(border_values));

    // segment 1
    segments.push_back(CreateSegment1(input_params, border_values));

    // It was decided based on 89164 to introduce extra segment to ensure that PWL for Identity will cross (0,0).
    // count zero of function for middle segment.
    //
    // Check if segment 1 passes passes throygh the point  (0,0)
    auto y0 = CountYAndValidateForX0(border_values, segments[1]);

    if (y0 != 0) {
        gnalog() << "PWL does not pass (0,0), F(0)=" << y0 << "! Adjusting PWL segments.";
        // if y0 != 0 add new segment, update previous one and cound properly next one if needed.

        // create a new segment with xBase = 0 and yBase = 0
        // use the same slope and scale as for previous segment
        segments.push_back(CreateSegment0_0(input_params));

        // adapt xBase for point on the left side of 0,0 to ensure that F(-1) <= F(0)
        UpdateSegmentOnTheLeftOf0_0(segments[0], segments[1], y0);
    }

    AddRightSegmentIFNeeded(border_values, segments);

    return segments;
}

void PWLSegmentsCreatorIdentity::AddRightSegmentIFNeeded(const ov::intel_gna::backend::BorderValues& border_values,
                                                         std::vector<gna_pwl_segment_t>& segments) const {
    if (std::numeric_limits<int32_t>::max() > border_values.x_upper) {
        auto back_segment = segments.back();
        segments.push_back(CreateSegmentOnTheRight(segments.back(), border_values));
    }
}

gna_pwl_segment_t PWLSegmentsCreatorIdentity::CreateSegment0(
    const ov::intel_gna::backend::BorderValues& border_values) const {
    return {ComputeXBaseForSegment(std::numeric_limits<int32_t>::min(), 0), border_values.y_lower, 0};
}

gna_pwl_segment_t PWLSegmentsCreatorIdentity::CreateSegment1(
    const PWLInputParams& input_params,
    const ov::intel_gna::backend::BorderValues& border_values) const {
    auto slope = ComputeSlopeForSegment(1.0, input_params.in_scale(), input_params.out_scale());
    int32_t x_base = ComputeXBaseForSegment(border_values.x_lower, slope.index);
    return {x_base, border_values.y_lower, slope.value};
}

gna_pwl_segment_t PWLSegmentsCreatorIdentity::CreateSegment0_0(const PWLInputParams& input_params) const {
    auto slope = ComputeSlopeForSegment(1.0, input_params.in_scale(), input_params.out_scale());
    auto x_base = ComputeXBaseForSegment(0, slope.index);
    return {x_base, 0, slope.value};
}

int16_t PWLSegmentsCreatorIdentity::CountYAndValidateForX0(const ov::intel_gna::backend::BorderValues& border_values,
                                                           const gna_pwl_segment_t& segment) const {
    auto y0 = ComputePWL(segment, 0);

    if (y0 > border_values.y_upper || y0 < border_values.y_lower) {
        THROW_GNA_EXCEPTION << "Invalid parameters. F(0)=" << y0 << " exceedes allowed values <"
                            << border_values.y_lower << ", " << border_values.y_upper << ">";
    }
    return static_cast<int16_t>(y0);
}

void PWLSegmentsCreatorIdentity::UpdateSegmentOnTheLeftOf0_0(const gna_pwl_segment_t& segment_0,
                                                             gna_pwl_segment_t& segment_1,
                                                             const int64_t delta_y) const {
    // new_segment_1_x_base = segment_1.xBase - (segment_0.yBase - segment_1.yBase - y_delta) * segment_1.scale_factor /
    // segment_1.slope
    auto segment_0_in_int64 = ConvertSegmentTo64(segment_0);
    auto segment_1_in_int64 = ConvertSegmentTo64(segment_1);

    if (segment_1_in_int64.slope == 0) {
        THROW_GNA_EXCEPTION << "Slope is 0 possible division by 0 when updating left segment!.";
    }

    int64_t new_segment_1_x_base =
        segment_1_in_int64.x_base - (segment_0_in_int64.y_base - segment_1_in_int64.y_base - delta_y) *
                                        segment_1_in_int64.slope_scale / segment_1_in_int64.slope;

    // to ensure that segment 1 will not have x_base lower than segment 0
    if (new_segment_1_x_base < segment_0_in_int64.x_base && new_segment_1_x_base + 1 < 0) {
        new_segment_1_x_base = segment_0_in_int64.x_base + 1;
    }

    new_segment_1_x_base = Round2LSBTowardZero(new_segment_1_x_base);
    auto slope_scale_index = GetScaleIndex(segment_1.xBase);
    segment_1.xBase = ComputeXBaseForSegment(static_cast<int32_t>(new_segment_1_x_base), slope_scale_index);
}

gna_pwl_segment_t PWLSegmentsCreatorIdentity::CreateSegmentOnTheRight(const gna_pwl_segment_t& segment_0_0,
                                                                      const BorderValues& border_values) const {
    // right_segment_x_base = segment_0_0.xBase + (border_values.y_upper - segment_0_0.yBase ) *
    // segment_0_0.scale_facotr / segment_0_0.slope
    auto segment_0_0_in_int64 = ConvertSegmentTo64(segment_0_0);
    int64_t right_y_base_64 = static_cast<int64_t>(border_values.y_upper);

    if (segment_0_0_in_int64.slope == 0) {
        THROW_GNA_EXCEPTION << "Slope is 0 possible division by 0 when calculating right segment!.";
    }

    int64_t right_segment_x_base = segment_0_0_in_int64.x_base + (right_y_base_64 - segment_0_0_in_int64.y_base) *
                                                                     segment_0_0_in_int64.slope_scale /
                                                                     segment_0_0_in_int64.slope;
    if (right_segment_x_base > std::numeric_limits<int32_t>::max()) {
        right_segment_x_base = std::numeric_limits<int32_t>::max();
    }

    right_segment_x_base = Round2LSBTowardZero(right_segment_x_base);
    auto segment_x_base_32 = ComputeXBaseForSegment(static_cast<int32_t>(right_segment_x_base), 0);
    return {segment_x_base_32, border_values.y_upper, 0};
}

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov