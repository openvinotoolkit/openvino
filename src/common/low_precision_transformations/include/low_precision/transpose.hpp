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
 * @brief TransposeTransformation propagates dequantization operations through Transpose operation.
 *
 * For more details about the transformation, refer to
 * [TransposeTransformation](@ref openvino_docs_OV_UG_lpt_TransposeTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API TransposeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("TransposeTransformation", "0", LayerTransformation);
    TransposeTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& op) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
