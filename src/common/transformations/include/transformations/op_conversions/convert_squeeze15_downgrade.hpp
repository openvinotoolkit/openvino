// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
/**
 * @ingroup ov_transformation_common_api
 * @brief Converts Squeeze v15 to Squeeze v0.
 */
class TRANSFORMATIONS_API ConvertSqueeze15ToSqueeze0 : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertSqueeze15ToSqueeze0");
    ConvertSqueeze15ToSqueeze0();
};

}  // namespace pass
}  // namespace ov
