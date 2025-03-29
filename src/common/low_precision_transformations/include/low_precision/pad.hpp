// Copyright (C) 2018-2025 Intel Corporation
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
 * @brief PadTransformation propagates dequantization operations through Pad operation.
 *
 * For more details about the transformation, refer to
 * [PadTransformation](@ref openvino_docs_OV_UG_lpt_PadTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API PadTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("PadTransformation", "0", LayerTransformation);
    PadTransformation(const Params& params = Params());
    bool transform(pattern::Matcher& m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& op) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
