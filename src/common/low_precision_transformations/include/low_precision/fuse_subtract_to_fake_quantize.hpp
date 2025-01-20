// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/fuse_elementwise_to_fake_quantize.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief FuseSubtractToFakeQuantizeTransformation fuses Subtract operation to FakeQuantize.
 *
 * For more details about the transformation, refer to
 * [FuseSubtractToFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FuseSubtractToFakeQuantizeTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API FuseSubtractToFakeQuantizeTransformation : public FuseElementwiseToFakeQuantizeTransformation {
public:
    OPENVINO_RTTI("FuseSubtractToFakeQuantizeTransformation", "0", FuseElementwiseToFakeQuantizeTransformation);
    FuseSubtractToFakeQuantizeTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
