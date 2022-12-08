// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pwl_border_values_counter.hpp"

class FakeQuantizeParams;
namespace ov {
namespace intel_gna {
namespace backend {

class BorderValuesCounterIdentity : public PWLBorderValuesCounter {
public:
    BorderValuesCounterIdentity() = default;
    ~BorderValuesCounterIdentity() override = default;
    BorderValues CreateBorderValues(const PWLInputParams& input_params) const override;

private:
    static BorderValues CreateBorderValues(const BorderValues& default_values, double in_scale, double out_scale);
    static BorderValues CreateBorderValuesWithFakeQuantize(const BorderValues& default_values,
                                                           const FakeQuantizeParams& fake_quantize_params,
                                                           double in_scale,
                                                           double out_scale);
    static BorderValues CreateDefaultBorderValues(bool is_low_precision);
    static bool AreValuesValid(const BorderValues& values);
};

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov