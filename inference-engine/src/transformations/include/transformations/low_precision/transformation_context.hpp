// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "transformations/low_precision/quantization_details.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API TransformationContext {
public:
    explicit TransformationContext(std::shared_ptr<Function> network);
    std::shared_ptr<Function> network;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
