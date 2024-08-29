// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief MarkDequantizationSubgraph marks dequantization subgraph, that is:
 *     Convert->Subtract(optional)->Multiply
 * in two ways:
 * - first Convert is marked with DisableConstantFolding attribute, also if Subtract is present
 *   and its second input is a Convert - that Convert is marked with DisableConstantFolding as well,
 * - Subtract and Multiply are marked with 'DequantizationNode' attribute
 */
class TRANSFORMATIONS_API MarkDequantizationSubgraph : public MatcherPass {
public:
    OPENVINO_RTTI("MarkDequantizationSubgraph", "0");
    MarkDequantizationSubgraph(const element::TypeVector& precisions,
                               const bool fold_subtract_const = false,
                               const bool disable_fold_multiply_const = false);
};
}  // namespace pass
}  // namespace ov
