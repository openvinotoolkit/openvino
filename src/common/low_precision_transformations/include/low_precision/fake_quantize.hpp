// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief FakeQuantizeTransformation fuses dequantization operations into FakeQuantize operation.
 *
 * For more details about the transformation, refer to
 * [FakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FakeQuantizeTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API FakeQuantizeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("FakeQuantizeTransformation", "0");
    FakeQuantizeTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

    static bool checkElementwise(const std::shared_ptr<Node>& eltwise);

    static std::shared_ptr<ov::opset1::FakeQuantize> fuseElementwise(
            TransformationContext& context,
            MatcherPass* matcherPass,
            const std::shared_ptr<ov::opset1::FakeQuantize>& fakeQuantize,
            const bool updatePrecisions);
};

} // namespace low_precision
} // namespace pass
} // namespace ov
