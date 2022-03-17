// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API CompressFloatConstantsImpl;
class TRANSFORMATIONS_API AddOldApiMapToParameters;
class TRANSFORMATIONS_API CompressFloatConstants;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief CompressFloatConstantsImpl transformation replaces FP32/FP64 Constants with FP16 ones.
 */
class ov::pass::CompressFloatConstantsImpl : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("CompressFloatConstantsImpl", "0");
    CompressFloatConstantsImpl();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief AddOldApiMapToParameters transformation adds OldApiMap to each float input to the model.
 */
class ov::pass::AddOldApiMapToParameters : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddOldApiMapToParameters", "0");
    AddOldApiMapToParameters();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief CompressFloatConstants transformation replaces FP32/FP64 Constants with FP16 ones.
 */
class ov::pass::CompressFloatConstants : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("CompressFloatConstants", "0");
    CompressFloatConstants() {
        add_matcher<ov::pass::CompressFloatConstantsImpl>();
        add_matcher<ov::pass::AddOldApiMapToParameters>();
    }
};
