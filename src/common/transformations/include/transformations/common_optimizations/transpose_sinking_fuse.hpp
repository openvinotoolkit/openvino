// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeSinkingFuse;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeSinkingFuse transformation eliminates 2 consecutive Transposes if they result in no changes to input or
 * fuses them to single Transpose if input gets changed
 */
class ov::pass::TransposeSinkingFuse : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeSinkingFuse", "0");
    TransposeSinkingFuse();
};