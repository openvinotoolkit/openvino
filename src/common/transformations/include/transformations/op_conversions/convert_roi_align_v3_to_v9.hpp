// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertROIAlign3To9;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertROIAlign3To9 converts v3::ROIAlign into v9::ROIAlign.
 */
class ov::pass::ConvertROIAlign3To9 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertROIAlign3To9", "0");
    ConvertROIAlign3To9();
};
