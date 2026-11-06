// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Runs 1D GroupConvolutionBackpropData as 2D with a unit trailing spatial axis.
 *
 * 1D grouped weights [G, C_in/G, C_out/G, K] are rank 4, the same rank as non-grouped 2D weights, and the GPU
 * primitive reads them as non-grouped: results were wrong for stride > 1 and stride 1 failed to execute.
 * The 2D grouped path is correct, so 1D is mapped onto it:
 *
 *  data [N, C, L] -> Unsqueeze(-1) -> [N, C, L, 1] --\
 *                                                     GroupConvolutionBackpropData (2D) -> Squeeze(-1) -> [N, C', L']
 *  weights [G, I, O, K] -> Unsqueeze(-1) ----------/
 *
 * strides / dilations get a trailing 1, pads / output_padding a trailing 0; an output_shape input gets a trailing 1.
 */
class ConvertGroupConvBackpropData1DTo2D : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupConvBackpropData1DTo2D");
    ConvertGroupConvBackpropData1DTo2D();
};

}  // namespace ov::intel_gpu
