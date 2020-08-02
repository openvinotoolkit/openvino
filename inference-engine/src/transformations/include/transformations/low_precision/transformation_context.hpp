// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/quantization_details.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API TransformationContext {
public:
    explicit TransformationContext(std::shared_ptr<Function> function);
    std::shared_ptr<Function> function;
    std::unordered_set<std::string> quantizedFakeQuantizeNames;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
