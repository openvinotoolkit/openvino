// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertROIAlign9To3;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertROIAlign9To3 converts v9::ROIAlign into v3::ROIAlign.
 */
class ngraph::pass::ConvertROIAlign9To3 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertROIAlign9To3", "0");
    ConvertROIAlign9To3();
};
