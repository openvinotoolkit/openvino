// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>



#include "layer_transformation.hpp"
#include "common/fake_quantize_dequantization.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief ConcatTransformation propagates dequantization operations through Concat operation.
 *
 * For more details about the transformation, refer to
 * [ConcatTransformation](@ref openvino_docs_OV_UG_lpt_ConcatTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API ConcatTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("ConcatTransformation", "0", LayerTransformation);
    ConcatTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    static bool isQuantizedStatic(const std::shared_ptr<const Node>& layer);
};

} // namespace low_precision
} // namespace pass
} // namespace ov
