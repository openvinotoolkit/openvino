// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertInterpolate1ToInterpolate4;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertInterpolate1ToInterpolate4 covert v0:interpolate into v4::Interpolate.
 */
class ov::pass::ConvertInterpolate1ToInterpolate4 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertInterpolate1ToInterpolate4", "0");
    ConvertInterpolate1ToInterpolate4();
};
