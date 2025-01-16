// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TransposeReshapeEliminationForMatmul;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief TransposeReshapeEliminationForMatmul transformation eliminates Transpose and Reshape which were created to
 * align input and output dimension ranks before second MatMul input and after MatMul output
 * (for example, after Einsum Decomposition inside TensorFlow 1 and OpenVINO EinsumDecomposition transformation)
 */
class ov::pass::TransposeReshapeEliminationForMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeReshapeEliminationForMatmul", "0");
    TransposeReshapeEliminationForMatmul();
};
