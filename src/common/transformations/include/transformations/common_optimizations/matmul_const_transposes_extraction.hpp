// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

/**
 * @ingroup ie_transformation_common_api
 * @brief Resolves transpose_b key from MatMul operation if corresponding input is constant or FakeQuantize by inserting
 * Transpose
 */

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MatMulConstTransposesExtraction : public MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MatMulConstTransposesExtraction();
};

}  // namespace pass
}  // namespace ngraph
