// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/model.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/serialize.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FlushFP32SubnormalsToZero;

}  // namespace pass
}  // namespace ov

/* @ingroup ov_transformation_common_api
 * @brief FlushFP32SubnormalsToZero flushes f32 subnormals to zero.
 * This is read/write expensive transformation, therefore should be run at offline phase.
 */
class ov::pass::FlushFP32SubnormalsToZero : public MatcherPass {
public:
    OPENVINO_RTTI("FlushFP32SubnormalsToZero", "0");
    FlushFP32SubnormalsToZero();
};
