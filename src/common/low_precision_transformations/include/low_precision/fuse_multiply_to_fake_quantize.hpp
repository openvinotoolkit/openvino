// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/fuse_elementwise_to_fake_quantize.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief FuseMultiplyToFakeQuantizeTransformation fuses Multiply operation to FakeQuantize.
 *
 * For more details about the transformation, refer to
 * [FuseMultiplyToFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FuseMultiplyToFakeQuantizeTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API FuseMultiplyToFakeQuantizeTransformation : public FuseElementwiseToFakeQuantizeTransformation {
public:
    OPENVINO_RTTI("FuseMultiplyToFakeQuantizeTransformation", "0");
    FuseMultiplyToFakeQuantizeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
