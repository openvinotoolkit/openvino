// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief FoldFakeQuantizeTransformation evaluate FakeQuantize operations.
 *
 * For more details about the transformation, refer to
 * [FoldFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_FoldFakeQuantizeTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API FoldFakeQuantizeTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("FoldFakeQuantizeTransformation", "0", LayerTransformation);
    FoldFakeQuantizeTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool isConstantOutput(std::shared_ptr<ov::Node> op) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
