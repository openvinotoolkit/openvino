// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSFuse;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TSFuse transformation eliminates 2 consecutive Transposes if they result in no changes to input
 * or fuses them to single Transpose if input gets changed
 */
class ov::pass::transpose_sinking::TSFuse : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TSFuse");
    TSFuse();
};
