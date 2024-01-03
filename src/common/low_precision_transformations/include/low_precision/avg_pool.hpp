// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief AvgPoolTransformation propagates dequantization operations through AvgPool operation.
 *
 * For more details about the transformation, refer to
 * [AvgPoolTransformation](@ref openvino_docs_OV_UG_lpt_AvgPoolTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API AvgPoolTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("AvgPoolTransformation", "0");
    AvgPoolTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
