// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief MaxPoolTransformation propagates dequantization operations through MaxPool operation.
 *
 * For more details about the transformation, refer to
 * [MaxPoolTransformation](@ref openvino_docs_OV_UG_lpt_MaxPoolTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API MaxPoolTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("MaxPoolTransformation", "0");
    MaxPoolTransformation(const Params& params = Params());
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
