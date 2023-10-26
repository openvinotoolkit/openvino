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
 * @brief ReshapeTransformation propagates dequantization operations through Reshape operation.
 *
 * For more details about the transformation, refer to
 * [ReshapeTransformation](@ref openvino_docs_OV_UG_lpt_ReshapeTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API ReshapeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("ReshapeTransformation", "0");
    ReshapeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;

    static bool canBeTransformed(
        const ov::Shape& subtractShape,
        const ov::Shape& multiplyShape,
        const ov::PartialShape& inputShape,
        const ov::PartialShape& outputShape);
};

} // namespace low_precision
} // namespace pass
} // namespace ov
