// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>
#include "low_precision/quantization_details.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief TransformationContext instance is used to pass model transformation context data between transformations.
 */
class LP_TRANSFORMATIONS_API TransformationContext {
public:
    TransformationContext();
    explicit TransformationContext(std::shared_ptr<Model> model);
    std::shared_ptr<Model> model;

    // Used to store handled FakeQuantize operations.
    // ConcatTransformation and FakeQuantizeTransformation handle FakeQuantize operations. ConcatTransformation handles FakeQuantize operation first.
    // If updatePrecision transformation option is set to False then there are no FakeQuantize operation attributes to identify that the operation
    // have been handled by ConcatTransformation already:
    //   - output precision is original (FP32),
    //   - intervals are changed but not equal to precision boundaries,
    //   - quantization level can be or can be not changed.
    // To avoid FakeQuantize operation double handling by FakeQuantizeTransformation after ConcatTransformation, FakeQuantizeTransformation
    // has to use this member.
    std::unordered_set<std::string> quantizedFakeQuantizeNames;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
