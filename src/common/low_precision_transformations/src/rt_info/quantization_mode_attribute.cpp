// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/quantization_mode_attribute.hpp"
#include <assert.h>
#include <sstream>

using namespace ov;
using namespace ov;

std::string QuantizationModeAttribute::to_string() const {
    assert((mode == Mode::Asymmetric) || (mode == Mode::Symmetric));

    std::stringstream ss;
    switch (mode) {
        case Mode::Asymmetric: {
            ss << "Asymmetric";
            break;
        }
        case Mode::Symmetric: {
            ss << "Symmetric";
            break;
        }
        default: {
            ss << "UNKNOWN";
            break;
        }
    }
    return ss.str();
}