// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertROIAlign9To3;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertROIAlign9To3 converts v9::ROIAlign into v3::ROIAlign.
 */
class ov::pass::ConvertROIAlign9To3 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertROIAlign9To3", "0");
    ConvertROIAlign9To3();
};
