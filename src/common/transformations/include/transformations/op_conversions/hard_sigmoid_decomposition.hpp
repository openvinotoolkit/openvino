// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API HardSigmoidDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief HardSigmoidDecomposition transformation into sub-graph.
 */
class ov::pass::HardSigmoidDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("HardSigmoidDecomposition", "0");
    HardSigmoidDecomposition();
};
