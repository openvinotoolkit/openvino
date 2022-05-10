// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertROIAlign3To9;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertROIAlign3To9 converts v3::ROIAlign into v9::ROIAlign.
 */
class ngraph::pass::ConvertROIAlign3To9 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertROIAlign3To9", "0");
    ConvertROIAlign3To9();
};
