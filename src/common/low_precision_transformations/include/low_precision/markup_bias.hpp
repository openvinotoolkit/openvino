// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "low_precision/lpt_visibility.hpp"
#include <memory>
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkupBias transformation marks biases after target layers.
 *
 * For more details about the transformation, refer to
 * [MarkupBias](@ref openvino_docs_OV_UG_lpt_MarkupBias) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API MarkupBias : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkupBias", "0");
    MarkupBias();
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov