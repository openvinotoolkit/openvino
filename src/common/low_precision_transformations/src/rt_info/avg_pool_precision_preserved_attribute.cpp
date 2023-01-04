// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"

#include <memory>
#include <vector>
#include <ngraph/variant.hpp>

using namespace ngraph;
using namespace ov;


void AvgPoolPrecisionPreservedAttribute::merge(std::vector<ov::Any>& attributes) {
}

bool AvgPoolPrecisionPreservedAttribute::is_skipped() const {
    return false;
}

std::string AvgPoolPrecisionPreservedAttribute::to_string() const {
    std::stringstream ss;
    ss << attribute->get_string();
    ss << "value: " << (value() ? "true" : "false");
    return ss.str();
}
