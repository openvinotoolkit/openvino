// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NormalizeL2Decomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Decomposes NormalizeL2 into subgraph
 */
class ov::pass::NormalizeL2Decomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NormalizeL2Decomposition", "0");
    NormalizeL2Decomposition();
};
