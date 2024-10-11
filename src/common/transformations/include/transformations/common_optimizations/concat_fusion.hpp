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

class TRANSFORMATIONS_API ConcatFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConcatFusion transformation fuses sequence of Concats
 */

class ov::pass::ConcatFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConcatFusion", "0");
    ConcatFusion();
};
