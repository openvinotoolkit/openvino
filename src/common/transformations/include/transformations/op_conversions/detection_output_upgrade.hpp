// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <openvino/core/ov_visibility.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API ConvertDetectionOutput1ToDetectionOutput8;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertDetectionOutput1ToDetectionOutput8 converts v0::DetectionOutput
 * into v8::DetectionOutput.
 */
class ngraph::pass::ConvertDetectionOutput1ToDetectionOutput8 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertDetectionOutput1ToDetectionOutput8();
};
