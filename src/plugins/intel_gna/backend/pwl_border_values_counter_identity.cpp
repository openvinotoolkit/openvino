// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_border_values_counter_identity.hpp"

#include "pwl_input_params.hpp"
#include "round_float_define.hpp"

namespace ov {
namespace intel_gna {
namespace backend {

BorderValues BorderValuesCounterIdentity::CreateBorderValues(const PWLInputParams& input_params) const {
    int32_t x_lower = INT32_MIN;
    int32_t x_upper = INT32_MAX;
    const bool is_low_precision = input_params.is_low_precision();
    const int16_t y_min = is_low_precision ? INT8_MIN : INT16_MIN;
    const int16_t y_max = is_low_precision ? INT8_MAX : INT16_MAX;
    int16_t y_lower = y_min;
    int16_t y_upper = y_max;

    const auto& fake_quantize_params = input_params.fake_quntize_params();
    auto in_scale = input_params.in_scale();
    auto out_scale = input_params.out_scale();
    if (static_cast<bool>(input_params.fake_quntize_params().set)) {
        x_lower = std::max(static_cast<const int64_t>(*fake_quantize_params.input_low * in_scale),
                           static_cast<int64_t>(x_lower));
        x_upper = std::min(static_cast<const int64_t>(*fake_quantize_params.input_high * in_scale),
                           static_cast<int64_t>(x_upper));
        y_lower = std::max(static_cast<const int32_t>(*fake_quantize_params.input_low * out_scale),
                           static_cast<int32_t>(y_lower));
        y_upper = std::min(static_cast<const int32_t>(*fake_quantize_params.input_high * out_scale),
                           static_cast<int32_t>(y_upper));
    } else {
        if (x_lower < y_lower * in_scale / out_scale)
            x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
        if (x_upper > y_upper * in_scale / out_scale)
            x_upper = FLOAT_TO_INT32(y_upper * in_scale / out_scale);
        if (y_lower < x_lower * out_scale / in_scale)
            y_lower = FLOAT_TO_INT16(x_lower * out_scale / in_scale);
        if (y_upper > x_upper * out_scale / in_scale)
            y_upper = FLOAT_TO_INT16(x_upper * out_scale / in_scale);
    }

    return {x_lower, x_upper, y_lower, y_upper, {y_min, y_max}};
}

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov