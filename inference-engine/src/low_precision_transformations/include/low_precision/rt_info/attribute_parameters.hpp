// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/type/element_type.hpp>
#include "low_precision/lpt_visibility.hpp"

class LP_TRANSFORMATIONS_API AttributeParameters {
public:
    AttributeParameters(const ov::element::Type deqPrecision = ov::element::f32) : deqPrecision(deqPrecision) {}
    ov::element::Type deqPrecision;
};
