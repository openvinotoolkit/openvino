// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "low_precision/lpt_visibility.hpp"

class LP_TRANSFORMATIONS_API AttributeParameters {
public:
    AttributeParameters(
        const ov::element::Type deqPrecision = ov::element::f32,
        const std::vector<ov::element::Type> defaultPrecisions = { ov::element::u8, ov::element::i8 })
    : deqPrecision(deqPrecision), defaultPrecisions(defaultPrecisions) {}
    ov::element::Type deqPrecision;
    std::vector<ov::element::Type> defaultPrecisions;
};
