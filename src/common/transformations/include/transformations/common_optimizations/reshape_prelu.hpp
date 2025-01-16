// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReshapePRelu;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ReshapePRelu reshape second input of PRelu (slope)
 */

class ov::pass::ReshapePRelu : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapePRelu", "0");
    ReshapePRelu();
};
