// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "backend/gna_types.h"
#include "pwl_border_values_counter.hpp"

namespace ov {
namespace intel_gna {
namespace backend {

class PWLInputParams;

struct PWLSegmentsWithBorderValues {
    BorderValues border_values;
    std::vector<gna_pwl_segment_t> segments;
};

class PWLSegmentsCreator {
public:
    virtual std::vector<gna_pwl_segment_t> CreateSegments(const PWLInputParams& input_params) const = 0;
    virtual PWLSegmentsWithBorderValues CreateSegmentsWithBorders(const PWLInputParams& input_params) const = 0;
    virtual ~PWLSegmentsCreator() = default;

private:
    std::shared_ptr<PWLBorderValuesCounter> counter_;
};

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov