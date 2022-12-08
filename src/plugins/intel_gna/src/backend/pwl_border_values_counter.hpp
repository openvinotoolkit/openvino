// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace ov {
namespace intel_gna {
namespace backend {

class PWLInputParams;

struct YMinMax {
    int16_t y_min;
    int16_t y_max;
};
struct BorderValues {
    int32_t x_lower;
    int32_t x_upper;
    int16_t y_lower;
    int16_t y_upper;
    YMinMax y_min_max;
};

class PWLBorderValuesCounter {
public:
    virtual ~PWLBorderValuesCounter() = default;
    virtual BorderValues CreateBorderValues(const PWLInputParams& input_params) const = 0;
};
}  // namespace backend
}  // namespace intel_gna
}  // namespace ov
