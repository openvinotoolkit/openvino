// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief ReshapeTransformation propagates dequantization operations through Reshape operation.
 *
 * For more details about the transformation, refer to
 * [ReshapeTransformation](@ref openvino_docs_OV_UG_lpt_ReshapeTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API ReshapeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("ReshapeTransformation", "0", LayerTransformation);
    ReshapeTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& op) const override;

    static bool canBeTransformed(
        const ov::Shape& subtractShape,
        const ov::Shape& multiplyShape,
        const ov::PartialShape& inputShape,
        const ov::PartialShape& outputShape);
};

} // namespace low_precision
} // namespace pass
} // namespace ov
