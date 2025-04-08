// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/matcher.hpp"
#include "low_precision/eltwise_base_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief MultiplyPartialTransformation propagates dequantization operations through Multiply operation.
 *
 * For more details about the transformation, refer to
 * [MultiplyPartialTransformation](@ref openvino_docs_OV_UG_lpt_MultiplyPartialTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API MultiplyPartialTransformation : public EltwiseBaseTransformation {
public:
    OPENVINO_RTTI("MultiplyPartialTransformation", "0", EltwiseBaseTransformation);
    MultiplyPartialTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
