// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/cleanup_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief FoldConvertTransformation evaluates Convert operation on Subtract constant subgraph.
 *
 * For more details about the transformation, refer to
 * [FoldConvertTransformation](@ref openvino_docs_OV_UG_lpt_FoldConvertTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API FoldConvertTransformation : public CleanupTransformation {
public:
    OPENVINO_RTTI("FoldConvertTransformation", "0");
    FoldConvertTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
