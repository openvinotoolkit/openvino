// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief SqueezeTransformation propagates dequantization operations through Squeeze operation.
 *
 * For more details about the transformation, refer to
 * [SqueezeTransformation](@ref openvino_docs_OV_UG_lpt_SqueezeTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API SqueezeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("SqueezeTransformation", "0", LayerTransformation);
    SqueezeTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
