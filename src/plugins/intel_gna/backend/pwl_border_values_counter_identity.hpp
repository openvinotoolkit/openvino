// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pwl_border_values_counter.hpp"

namespace ov {
namespace intel_gna {
namespace backend {

class BorderValuesCounterIdentity : public PWLBorderValuesCounter {
public:
    BorderValuesCounterIdentity() = default;
    ~BorderValuesCounterIdentity() override = default;
    BorderValues CreateBorderValues(const PWLInputParams& input_params) const override;
};

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov