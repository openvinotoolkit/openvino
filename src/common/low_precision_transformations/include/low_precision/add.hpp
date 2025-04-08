// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "low_precision/eltwise_base_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief AddTransformation propagates dequantization subtraction from one input branch to another and
 * propagates dequantization multiplication from the same branch through Add operation.
 *
 * For more details about the transformation, refer to
 * [AddTransformation](@ref openvino_docs_OV_UG_lpt_AddTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API AddTransformation : public EltwiseBaseTransformation {
public:
    OPENVINO_RTTI("AddTransformation", "0", EltwiseBaseTransformation);
    AddTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
