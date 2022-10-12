// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/dnn_types.h"

namespace ov {
namespace intel_gna {
namespace backend {

class PWLInputParams {
public:
    PWLInputParams(bool is_low_precision,
                   const FakeQuantizeParams& fake_qunatize_params,
                   double in_scale,
                   double out_scale);

    bool is_low_precision() const;
    const FakeQuantizeParams& fake_quntize_params() const;
    double in_scale() const;
    double out_scale() const;

private:
    bool is_low_precision_;
    FakeQuantizeParams fake_quntize_params_;
    double in_scale_;
    double out_scale_;
};
}  // namespace backend
}  // namespace intel_gna
}  // namespace ov