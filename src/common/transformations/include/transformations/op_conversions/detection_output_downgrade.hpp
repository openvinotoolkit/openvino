// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertDetectionOutput8ToDetectionOutput1;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertDetectionOutput8ToDetectionOutput1 converts v8::DetectionOutput
 * into v0::DetectionOutput.
 */
class ngraph::pass::ConvertDetectionOutput8ToDetectionOutput1 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertDetectionOutput8ToDetectionOutput1", "0");
    ConvertDetectionOutput8ToDetectionOutput1();
};
