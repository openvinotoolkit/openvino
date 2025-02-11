// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/cleanup_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief FoldConvertTransformation evaluates Convert operation on Subtract constant subgraph.
 * Important notice: this transformation ignores DisableConstantFolding runtime attribute.
 *
 * For more details about the transformation, refer to
 * [FoldConvertTransformation](@ref openvino_docs_OV_UG_lpt_FoldConvertTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API FoldConvertTransformation : public CleanupTransformation {
public:
    OPENVINO_RTTI("FoldConvertTransformation", "0", CleanupTransformation);
    FoldConvertTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
