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
 * @brief PReluTransformation propagates dequantization operations through PRelu operation.
 *
 * For more details about the transformation, refer to
 * [PReluTransformation](@ref openvino_docs_OV_UG_lpt_PReluTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API PReluTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("PReluTransformation", "0");
    PReluTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
