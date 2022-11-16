// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_input_params.hpp"

namespace ov {
namespace intel_gna {
namespace backend {

PWLInputParams::PWLInputParams(bool is_low_precision,
                               const FakeQuantizeParams& fake_qunatize_params,
                               double in_scale,
                               double out_scale)
    : is_low_precision_(is_low_precision),
      fake_quntize_params_(fake_qunatize_params),
      in_scale_(in_scale),
      out_scale_(out_scale) {}

bool PWLInputParams::is_low_precision() const {
    return is_low_precision_;
}

const FakeQuantizeParams& PWLInputParams::fake_quntize_params() const {
    return fake_quntize_params_;
}

double PWLInputParams::in_scale() const {
    return in_scale_;
}

double PWLInputParams::out_scale() const {
    return out_scale_;
}

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov