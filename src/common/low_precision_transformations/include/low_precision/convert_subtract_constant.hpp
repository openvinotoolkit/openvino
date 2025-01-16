// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "low_precision/lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ConvertSubtractConstant;

}  // namespace low_precision
}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertSubtractConstant marks Convert operations on constant subgraph by DISABLED_CONSTANT_FOLDING attribute
 * to prevent constant folding.
 *
 * For more details about the transformation, refer to
 * [ConvertSubtractConstant](@ref openvino_docs_OV_UG_lpt_ConvertSubtractConstant) page
 * in the OpenVINO Developer Guide.
 */
class ov::pass::low_precision::ConvertSubtractConstant : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertSubtractConstant", "0");
    ConvertSubtractConstant(const std::vector<ov::element::Type>& constantPrecisions = {});
};
