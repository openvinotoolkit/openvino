// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

/**
 * @ingroup ov_transformation_common_api
 * @brief Resolves transpose_b key from MatMul operation if corresponding input is constant or FakeQuantize by inserting
 * Transpose
 */

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MatMulConstTransposesExtraction : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MatMulConstTransposesExtraction");
    MatMulConstTransposesExtraction();
};

}  // namespace pass
}  // namespace ov
