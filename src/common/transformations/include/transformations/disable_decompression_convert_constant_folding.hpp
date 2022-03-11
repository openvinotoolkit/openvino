// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DisableDecompressionConvertConstantFolding;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Disables ConstantFolding for Convert operation in compressed function.
 */
class ov::pass::DisableDecompressionConvertConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DisableDecompressionConvertConstantFolding", "0");
    DisableDecompressionConvertConstantFolding();
};
