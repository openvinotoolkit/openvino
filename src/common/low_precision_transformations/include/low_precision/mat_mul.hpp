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
 * @brief MatMulTransformation propagates dequantization operations through MatMul operation.
 *
 * For more details about the transformation, refer to
 * [MatMulTransformation](@ref openvino_docs_OV_UG_lpt_MatMulTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API MatMulTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("MatMulTransformation", "0", LayerTransformation);
    MatMulTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
