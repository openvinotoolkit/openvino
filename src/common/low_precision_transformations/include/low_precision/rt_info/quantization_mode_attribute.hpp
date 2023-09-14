// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "openvino/core/runtime_attribute.hpp"
#include "low_precision/lpt_visibility.hpp"

namespace ov {

class LP_TRANSFORMATIONS_API QuantizationModeAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("LowPrecision::QuantizationModeAttribute", "", ov::RuntimeAttribute);

    enum class Mode {
        Asymmetric,
        Symmetric
    };

    QuantizationModeAttribute() : mode(Mode::Asymmetric) {}
    QuantizationModeAttribute(const Mode mode) : mode(mode) {}

    bool operator==(const QuantizationModeAttribute& attribute) const {
        return this->mode == attribute.mode;
    }

    std::string to_string() const override;

    Mode mode;
};
} // namespace ov
