// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertXorToLogicalXor;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertXorToLogicalXor converts v0::Xor to v1::LogicalXor.
 */
class ov::pass::ConvertXorToLogicalXor : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertXorToLogicalXor", "0");
    ConvertXorToLogicalXor();
};
