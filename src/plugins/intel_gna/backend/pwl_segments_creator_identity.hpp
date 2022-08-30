// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pwl_segments_creator.hpp"

enum DnnActivationType : uint8_t;

namespace ov {
namespace intel_gna {
namespace backend {

class PWLSegmentsCreatorIdentity : public PWLSegmentsCreator {
public:
    PWLSegmentsCreatorIdentity(std::shared_ptr<PWLBorderValuesCounter> border_counter);
    ~PWLSegmentsCreatorIdentity() override = default;

protected:
    std::vector<gna_pwl_segment_t> CreateSegments(const PWLInputParams& input_params) const override;
    PWLSegmentsWithBorderValues CreateSegmentsWithBorders(const PWLInputParams& input_params) const override;

private:
    std::vector<gna_pwl_segment_t> CreateSegments(const PWLInputParams& input_params,
                                                  const BorderValues& border_values) const;
    void UpdateSegmentOnTheLeftOf0_0(const gna_pwl_segment_t& before_left_segment,
                                     gna_pwl_segment_t& left_segment,
                                     const int64_t delta_y,
                                     uint32_t slope_scale_index) const;
    gna_pwl_segment_t CalculcateLastSegmentOnTheRightOf0_0(const gna_pwl_segment_t& segment_0_0,
                                                           const BorderValues& border_values) const;

private:
    std::shared_ptr<PWLBorderValuesCounter> border_counter_;
};

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov