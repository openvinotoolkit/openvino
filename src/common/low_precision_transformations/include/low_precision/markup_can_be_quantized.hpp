// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/pass/pass.hpp"
#include "low_precision/lpt_visibility.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API MarkupCanBeQuantized;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief MarkupCanBeQuantized transformation marks Convolution, ConvolutionBackpropData, GroupConvolution and Concat
 * operations as able to be quantized or not. If an operation is not quantized, then PrecisionsAttribute attribute instance
 * is created with empty precisions.
 *
 * For more details about the transformation, refer to
 * [MarkupCanBeQuantized](@ref openvino_docs_OV_UG_lpt_MarkupCanBeQuantized) page
 * in the OpenVINO Developer Guide.
 */
class ov::pass::low_precision::MarkupCanBeQuantized : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("low_precision::MarkupCanBeQuantized");
    MarkupCanBeQuantized(const std::vector<ov::element::Type> defaultPrecisions = { ov::element::u8, ov::element::i8 });
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
private:
    const std::vector<ov::element::Type> defaultPrecisions;
};
