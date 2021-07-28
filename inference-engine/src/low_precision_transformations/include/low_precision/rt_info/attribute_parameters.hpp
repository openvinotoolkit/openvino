// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/type/element_type.hpp>
#include "low_precision/lpt_visibility.hpp"

class LP_TRANSFORMATIONS_API AttributeParameters {
public:
    AttributeParameters(const ngraph::element::Type deqPrecision = ngraph::element::f32) : deqPrecision(deqPrecision) {}
    ngraph::element::Type deqPrecision;
};
