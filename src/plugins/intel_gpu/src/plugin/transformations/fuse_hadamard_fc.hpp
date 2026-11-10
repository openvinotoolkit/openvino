// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_gpu {

// rt_info keys carried on a FullyConnectedCompressed whose input transform was
// absorbed; the FC translator moves them onto the cldnn primitive.
inline constexpr char int2_hadamard_block_key[] = "int2_hadamard_block";
inline constexpr char int2_hadamard_signs_key[] = "int2_hadamard_signs";

// Folds the graph-level input rotation of Hadamard-basis checkpoints (Bonsai 2)
//
//   x -> [Multiply(+-1 signs)] -> Reshape(..., K/1024, 1024) -> MatMul(H_1024)
//     -> Reshape(..., K) -> FullyConnectedCompressed(u2)
//
// into the FullyConnected itself, which then runs one fused sign+FWHT kernel in
// front of its GEMV instead of three graph primitives. Only the TernOCL int2 FC
// implementation honours the resulting primitive fields.
class FuseHadamardIntoFC : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseHadamardIntoFC");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_gpu
