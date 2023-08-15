// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_border_values_counter_identity.hpp"

#include <limits>

#include "common/numerical_utils.hpp"
#include "log/debug.hpp"
#include "pwl_input_params.hpp"

namespace ov {
namespace intel_gna {
namespace backend {

BorderValues BorderValuesCounterIdentity::CreateBorderValues(const PWLInputParams& input_params) const {
    auto results = CreateDefaultBorderValues(input_params.is_low_precision());

    const auto& fake_quantize_params = input_params.fake_quntize_params();
    auto in_scale = input_params.in_scale();
    auto out_scale = input_params.out_scale();
    if (static_cast<bool>(input_params.fake_quntize_params().set)) {
        results = CreateBorderValuesWithFakeQuantize(results, fake_quantize_params, in_scale, out_scale);
    } else {
        results = CreateBorderValues(results, in_scale, out_scale);
    }
    if (!AreValuesValid(results)) {
        THROW_GNA_EXCEPTION << "Unacceptable scale factors. Difference between factors is too big: in_scale="
                            << in_scale << ", out_scale: " << out_scale;
    }

    return results;
}

BorderValues BorderValuesCounterIdentity::CreateBorderValues(const BorderValues& default_values,
                                                             double in_scale,
                                                             double out_scale) {
    int32_t x_lower = default_values.x_lower;
    int32_t x_upper = default_values.x_upper;
    int16_t y_lower = default_values.y_lower;
    int16_t y_upper = default_values.y_upper;
    if (x_lower < y_lower * in_scale / out_scale)
        x_lower = common::DoubleToInt32(y_lower * in_scale / out_scale);
    if (x_upper > y_upper * in_scale / out_scale)
        x_upper = common::DoubleToInt32(y_upper * in_scale / out_scale);
    if (y_lower < x_lower * out_scale / in_scale)
        y_lower = common::DoubleToInt16(x_lower * out_scale / in_scale);
    if (y_upper > x_upper * out_scale / in_scale)
        y_upper = common::DoubleToInt16(x_upper * out_scale / in_scale);
    return {x_lower, x_upper, y_lower, y_upper, {default_values.y_lower, default_values.y_upper}};
}

BorderValues BorderValuesCounterIdentity::CreateBorderValuesWithFakeQuantize(
    const BorderValues& default_values,
    const FakeQuantizeParams& fake_quantize_params,
    double in_scale,
    double out_scale) {
    int32_t x_lower =
        static_cast<int32_t>(std::max(static_cast<const int64_t>(*fake_quantize_params.input_low * in_scale),
                                      static_cast<int64_t>(default_values.x_lower)));
    int32_t x_upper =
        static_cast<int32_t>(std::min(static_cast<const int64_t>(*fake_quantize_params.input_high * in_scale),
                                      static_cast<int64_t>(default_values.x_upper)));
    int16_t y_lower = std::max(static_cast<const int32_t>(*fake_quantize_params.input_low * out_scale),
                               static_cast<int32_t>(default_values.y_lower));
    int16_t y_upper = std::min(static_cast<const int32_t>(*fake_quantize_params.input_high * out_scale),
                               static_cast<int32_t>(default_values.y_upper));

    return {x_lower, x_upper, y_lower, y_upper, {default_values.y_lower, default_values.y_upper}};
}

BorderValues BorderValuesCounterIdentity::CreateDefaultBorderValues(bool is_low_precision) {
    const int16_t y_min = is_low_precision ? std::numeric_limits<int8_t>::min() : std::numeric_limits<int16_t>::min();
    const int16_t y_max = is_low_precision ? std::numeric_limits<int8_t>::max() : std::numeric_limits<int16_t>::max();
    return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), y_min, y_max, {y_min, y_max}};
}

bool BorderValuesCounterIdentity::AreValuesValid(const BorderValues& values) {
    if (values.x_lower == values.x_upper && values.x_upper == values.y_lower && values.y_lower == values.y_upper &&
        values.y_upper == 0) {
        return false;
    }
    return true;
}
}  // namespace backend
}  // namespace intel_gna
}  // namespace ov
