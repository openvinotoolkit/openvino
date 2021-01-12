// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <ngraph/node.hpp>
#include <low_precision/lpt_visibility.hpp>

class LP_TRANSFORMATIONS_API AttributeParameters {
public:
    AttributeParameters(ngraph::element::Type deqPrecision = ngraph::element::f32) : deqPrecision(deqPrecision) {}
    ngraph::element::Type deqPrecision;
};
