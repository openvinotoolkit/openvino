// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/type/element_type.hpp>
#include "openvino/core/ov_visibility.hpp"

class OPENVINO_API AttributeParameters {
public:
    AttributeParameters(const ngraph::element::Type deqPrecision = ngraph::element::f32) : deqPrecision(deqPrecision) {}
    ngraph::element::Type deqPrecision;
};
