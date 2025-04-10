// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief SliceTransformation propagates dequantization operations through Slice operation.
 *
 * For more details about the transformation, refer to
 * [SliceTransformation](@ref openvino_docs_OV_UG_lpt_SliceTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API SliceTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("SliceTransformation", "0", LayerTransformation);
    SliceTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher& m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& op) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
