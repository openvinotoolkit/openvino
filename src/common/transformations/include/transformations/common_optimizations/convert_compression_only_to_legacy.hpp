// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EnableDecompressionConvertConstantFolding;
class TRANSFORMATIONS_API ConvertCompressedOnlyToLegacy;

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
 * @brief ConvertCompressedOnlyToLegacy transformation converts compression only FP16 format to legacy FP16 format.
 */
class ov::pass::ConvertCompressedOnlyToLegacy : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertCompressedOnlyToLegacy", "0");
    bool run_on_model(const std::shared_ptr<Model>& f) override;
};
