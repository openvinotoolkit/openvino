// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NonZeroHorizontalFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief NonZeroHorizontalFusion transformation makes horizontal fusion for equal NonZero layers
 */
class ov::pass::NonZeroHorizontalFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("NonZeroHorizontalFusion", "0");
    NonZeroHorizontalFusion();
};
