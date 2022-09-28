// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
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
    MarkDequantizationSubgraph(const element::TypeVector& precisions = {});
};
}  // namespace pass
}  // namespace ngraph
