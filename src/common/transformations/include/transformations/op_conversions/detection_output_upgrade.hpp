// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertDetectionOutput1ToDetectionOutput8;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertDetectionOutput1ToDetectionOutput8 converts v0::DetectionOutput
 * into v8::DetectionOutput.
 */
class ngraph::pass::ConvertDetectionOutput1ToDetectionOutput8 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDetectionOutput1ToDetectionOutput8", "0");
    ConvertDetectionOutput1ToDetectionOutput8();
};
