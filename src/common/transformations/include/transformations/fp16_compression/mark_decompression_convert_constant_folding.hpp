// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mark_decompression_convert_constant_folding.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EnableDecompressionConvertConstantFolding;
class TRANSFORMATIONS_API DisableDecompressionConvertConstantFolding;
class TRANSFORMATIONS_API KeepConstAndDecompression;
class TRANSFORMATIONS_API KeepConstantsPrecisionAndAddConverts;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Enables ConstantFolding for Convert operation in compressed function.
 */
class ov::pass::EnableDecompressionConvertConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EnableDecompressionConvertConstantFolding", "0");
    EnableDecompressionConvertConstantFolding();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Disables ConstantFolding for Convert operation in compressed function.
 */
class ov::pass::DisableDecompressionConvertConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DisableDecompressionConvertConstantFolding", "0");
    DisableDecompressionConvertConstantFolding();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Disables ConstantFolding for Convert operation and prevents conversion of f16 Consts to f32.
 */
class ov::pass::KeepConstAndDecompression : public MatcherPass {
public:
    OPENVINO_RTTI("KeepConstAndDecompression", "0");
    KeepConstAndDecompression();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Prevents Consts precision conversion and adds Convert with disabled ConstantFolding
 */
class ov::pass::KeepConstantsPrecisionAndAddConverts : public MatcherPass {
public:
    OPENVINO_RTTI("KeepConstantsPrecisionAndAddConverts", "0");
    KeepConstantsPrecisionAndAddConverts();
};
