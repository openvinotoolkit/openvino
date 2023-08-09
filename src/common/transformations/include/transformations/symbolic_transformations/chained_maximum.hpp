// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
class TRANSFORMATIONS_API ChainedMaximumOptimization;
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Optimizes graphs like Maximum(Maximum(A, B), B) to Maximum(A, B) based on value labels / symbols
 */
class ov::pass::ChainedMaximumOptimization : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ChainedMaximumOptimization", "0");
    ChainedMaximumOptimization();
};