// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "low_precision/weightable_layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief MultiplyTransformation propagates dequantization operations through Multiply operation.
 *
 * For more details about the transformation, refer to
 * [MultiplyTransformation](@ref openvino_docs_OV_UG_lpt_MultiplyTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API MultiplyTransformation : public WeightableLayerTransformation {
public:
    OPENVINO_RTTI("MultiplyTransformation", "0", WeightableLayerTransformation);
    MultiplyTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;

protected:
    size_t getInputChannels(const std::shared_ptr<ov::Node> op) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
