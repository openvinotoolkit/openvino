// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSResetNoSinkingAttribute;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief TSResetNoSinkingAttribute transformation resets all NoSinkingAttribute runtime attributes
 * in Transpose operations.
 */
class ov::pass::transpose_sinking::TSResetNoSinkingAttribute : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSResetNoSinkingAttribute", "0");
    TSResetNoSinkingAttribute();
};
