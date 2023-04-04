// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations/enable_decompression_convert_constant_folding.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EnableDecompressionConvertConstantFolding;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Disables ConstantFolding for Convert operation in compressed function.
 */
class ov::pass::EnableDecompressionConvertConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EnableDecompressionConvertConstantFolding", "0");
    EnableDecompressionConvertConstantFolding();
};
