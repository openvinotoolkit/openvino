// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ChainedMaximumOptimization;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Optimizes graphs based on value symbols
 *      Maximum(Maximum(A, B), B) -> Maximum(A, B)
 *      Maximum(Maximum(A, B), A) -> Maximum(A, B)
 */
class ov::pass::ChainedMaximumOptimization : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ChainedMaximumOptimization");
    ChainedMaximumOptimization();
};
