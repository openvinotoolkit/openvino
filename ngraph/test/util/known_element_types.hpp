// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/ngraph.hpp"

static const std::vector<ngraph::element::Type> s_known_element_types = {
    ngraph::element::from<float>(),
    ngraph::element::from<double>(),
    ngraph::element::from<int8_t>(),
    ngraph::element::from<int16_t>(),
    ngraph::element::from<int32_t>(),
    ngraph::element::from<int64_t>(),
    ngraph::element::from<uint8_t>(),
    ngraph::element::from<uint16_t>(),
    ngraph::element::from<uint32_t>(),
    ngraph::element::from<uint64_t>(),
};
