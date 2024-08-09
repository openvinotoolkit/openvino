// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "mat_mul.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief MatMulWithDequantizationTransformation propagates dequantization operations through MatMul operation and keep dequantisation as is.
 *
 * For more details about the transformation, refer to
 * [MatMulWithDequantizationTransformation](@ref openvino_docs_OV_UG_lpt_MatMulWithDequantizationTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API MatMulWithDequantizationTransformation : public MatMulTransformation {
public:
    OPENVINO_RTTI("MatMulWithDequantizationTransformation", "0");
    MatMulWithDequantizationTransformation(const Params& params = Params());

protected:
    void handleDequantization(const std::shared_ptr<ov::opset1::Multiply>& dequantization) const override;
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
