// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/ngraph.hpp"

static const std::vector<ov::element::Type> s_known_element_types = {
    ov::element::from<float>(),
    ov::element::from<double>(),
    ov::element::from<int8_t>(),
    ov::element::from<int16_t>(),
    ov::element::from<int32_t>(),
    ov::element::from<int64_t>(),
    ov::element::from<uint8_t>(),
    ov::element::from<uint16_t>(),
    ov::element::from<uint32_t>(),
    ov::element::from<uint64_t>(),
};
