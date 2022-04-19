// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <ngraph/variant.hpp>
#include <low_precision/lpt_visibility.hpp>

namespace ngraph {

class LP_TRANSFORMATIONS_API QuantizationModeAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("LowPrecision::QuantizationModeAttribute", "", ov::RuntimeAttribute, 0);

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
} // namespace ngraph
