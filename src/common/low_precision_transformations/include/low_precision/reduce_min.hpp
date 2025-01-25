// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "low_precision/reduce_base_transformation.hpp"

#include <memory>

#include "layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief ReduceMinTransformation propagates dequantization operations through ReduceMin operation.
 *
 * For more details about the transformation, refer to
 * [ReduceMinTransformation](@ref openvino_docs_OV_UG_lpt_ReduceMinTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API ReduceMinTransformation : public ReduceBaseTransformation {
public:
    OPENVINO_RTTI("ReduceMinTransformation", "0", ReduceBaseTransformation);
    ReduceMinTransformation(const Params& params = Params());
    bool isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& reduce) const override;

protected:
    bool getUpdatePrecision(const std::shared_ptr<Node>& reduce) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
