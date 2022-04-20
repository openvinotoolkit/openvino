// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ngraph/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API TransposeReshapeEliminationForMatmul;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeReshapeEliminationForMatmul transformation eliminates Transpose and Reshape which were created to
 * align input and output dimension ranks before second MatMul input and after MatMul output
 * (for example, after Einsum Decomposition inside TensorFlow 1 and nGraph EinsumDecomposition transformation)
 */
class ngraph::pass::TransposeReshapeEliminationForMatmul : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeReshapeEliminationForMatmul", "0");
    TransposeReshapeEliminationForMatmul();
};
