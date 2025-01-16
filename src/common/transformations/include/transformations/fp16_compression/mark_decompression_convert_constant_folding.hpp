// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mark_decompression_convert_constant_folding.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EnableDecompressionConvertConstantFolding;
class TRANSFORMATIONS_API DisableDecompressionConvertConstantFolding;
class TRANSFORMATIONS_API KeepConstAndDecompression;
class TRANSFORMATIONS_API KeepConstantsPrecisionAndAddConverts;
class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Enables ConstantFolding for Convert operation in compressed function.
 */
class ov::pass::EnableDecompressionConvertConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EnableDecompressionConvertConstantFolding", "0");
    EnableDecompressionConvertConstantFolding();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Disables ConstantFolding for Convert operation in compressed function.
 */
class ov::pass::DisableDecompressionConvertConstantFolding : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DisableDecompressionConvertConstantFolding", "0");
    DisableDecompressionConvertConstantFolding();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Disables ConstantFolding for Convert operation and prevents conversion of f16 Consts to f32.
 */
class ov::pass::KeepConstAndDecompression : public MatcherPass {
public:
    OPENVINO_RTTI("KeepConstAndDecompression", "0");
    KeepConstAndDecompression();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Prevents Consts precision conversion and adds Convert with disabled ConstantFolding
 */
class ov::pass::KeepConstantsPrecisionAndAddConverts : public MatcherPass {
public:
    OPENVINO_RTTI("KeepConstantsPrecisionAndAddConverts", "0");
    KeepConstantsPrecisionAndAddConverts();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Prevents ConstantFolding for f16/bf16 Const + Convert_To_FP32 to keep original FW float Constants.
 * Original precision should be kept as long as possible, this prevents redundant conversions and saves memory.
 * E.g. if original FW model was already compressed no need to upcast during CF, store intermediate f32 consts and
 * then again compress them to f16 during save_model.
 */
class ov::pass::MarkCompressedFloatConstants : public MatcherPass {
public:
    OPENVINO_RTTI("KeepFWPrecisionFor16BitFloatConstants", "0");
    MarkCompressedFloatConstants();
};
