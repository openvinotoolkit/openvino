// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace FuncTestUtils {
namespace PartialShapeUtils {

inline ngraph::PartialShape vec2partialshape(std::vector<std::pair<size_t, size_t>> inputShape, const ngraph::Shape& targetShape) {
    if (inputShape.empty()) {
        for (auto&& item : targetShape) {
            inputShape.emplace_back(item, item);
        }
    }
    std::vector<ngraph::Dimension> dimensions;
    dimensions.reserve(inputShape.size());
    for (auto&& item : inputShape) {
        dimensions.emplace_back(item.first, item.second);
    }
    return ngraph::PartialShape(dimensions);
}

}  // namespace PartialShapeUtils
}  // namespace FuncTestUtils
