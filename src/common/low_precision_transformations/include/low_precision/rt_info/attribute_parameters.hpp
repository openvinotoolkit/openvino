// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/type/element_type.hpp>
#include "low_precision/lpt_visibility.hpp"

class LP_TRANSFORMATIONS_API AttributeParameters {
public:
    AttributeParameters(
        const ngraph::element::Type deqPrecision = ngraph::element::f32,
        const std::vector<ngraph::element::Type> defaultPrecisions = { ngraph::element::u8, ngraph::element::i8 })
    : deqPrecision(deqPrecision), defaultPrecisions(defaultPrecisions) {}
    ngraph::element::Type deqPrecision;
    std::vector<ngraph::element::Type> defaultPrecisions;
};
