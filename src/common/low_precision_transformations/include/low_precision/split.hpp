// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "layer_transformation.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ie_transformation_common_api
 * @brief SplitTransformation propagates dequantization operations through Split operation.
 *
 * For more details about the transformation, refer to
 * [SplitTransformation](@ref openvino_docs_OV_UG_lpt_SplitTransformation) page
 * in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API SplitTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("SplitTransformation", "0");
    SplitTransformation(const Params& params = Params());
    bool transform(TransformationContext& context, ov::pass::pattern::Matcher& m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const override;
    void updateOutputs(
        TransformationContext& context,
        std::vector<std::shared_ptr<ov::Node>> lastNodes,
        std::shared_ptr<ov::Node> originalNode) const;
};
} // namespace low_precision
} // namespace pass
} // namespace ov
