// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/cleanup_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief EliminateFakeQuantizeTransformation removes FakeQuantize operations.
 *
 * For more details about the transformation, refer to
 * [EliminateFakeQuantizeTransformation](@ref openvino_docs_OV_UG_lpt_EliminateFakeQuantizeTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API EliminateFakeQuantizeTransformation : public CleanupTransformation {
public:
    OPENVINO_RTTI("EliminateFakeQuantizeTransformation", "0", CleanupTransformation);
    EliminateFakeQuantizeTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
