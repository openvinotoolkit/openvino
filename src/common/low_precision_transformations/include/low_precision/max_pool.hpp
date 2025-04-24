// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief MaxPoolTransformation propagates dequantization operations through MaxPool operation.
 *
 * For more details about the transformation, refer to
 * [MaxPoolTransformation](@ref openvino_docs_OV_UG_lpt_MaxPoolTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API MaxPoolTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("MaxPoolTransformation", "0", LayerTransformation);
    MaxPoolTransformation(const Params& params = Params());
    bool canBeTransformed(const std::shared_ptr<Node>& op) const override;
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
