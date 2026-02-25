// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul.hpp"

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

Result GatherMatmulShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                     [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    // Expected inputs:
    // 0: A - activations [group_size, M, K] (3D)
    // 1: B - weights [gather_axis, N, K] (3D)
    // 2: indices - gather indices [M, group_size] (2D)
    // Optional: 3: bias

    OPENVINO_DEBUG_ASSERT(input_shapes.size() >= 3,
                          "GatherMatmul shape inference expects at least 3 inputs, got: ",
                          input_shapes.size());

    const VectorDims& shapeA = input_shapes[0].get();
    const VectorDims& shapeB = input_shapes[1].get();
    const VectorDims& shapeIndices = input_shapes[2].get();

    // Validate input ranks
    OPENVINO_DEBUG_ASSERT(shapeA.size() == 3,
                          "GatherMatmul input A must be 3D, got rank: ",
                          shapeA.size(),
                          " with shape: ",
                          vec2str(shapeA));
    OPENVINO_DEBUG_ASSERT(shapeB.size() == 3,
                          "GatherMatmul input B must be 3D, got rank: ",
                          shapeB.size(),
                          " with shape: ",
                          vec2str(shapeB));
    OPENVINO_DEBUG_ASSERT(shapeIndices.size() == 2,
                          "GatherMatmul indices must be 2D, got rank: ",
                          shapeIndices.size(),
                          " with shape: ",
                          vec2str(shapeIndices));

    // Extract dimensions for matmul computation
    // A shape: [group, M, K]
    // B shape: [gather_axis, N, K]
    // We perform matmul on A[1:2] x B[1:2].T

    const size_t M = shapeA[1];
    const size_t K_A = shapeA[2];
    const size_t K = shapeB[1];
    const size_t N = shapeB[2];

    // Validate K-dimension compatibility

    const size_t k_lhs = m_transpose_a ? M : K_A;
    const size_t k_rhs = m_transpose_b ? N : K;

    OPENVINO_ASSERT(k_lhs == k_rhs,
                    "GatherMatmul: incompatible matmul dimensions. ",
                    "A shape: ",
                    vec2str(shapeA),
                    m_transpose_a ? " (transposed)" : "",
                    ", B shape: ",
                    vec2str(shapeB),
                    m_transpose_b ? " (transposed)" : "",
                    ". K dimension mismatch: ",
                    k_lhs,
                    " vs ",
                    k_rhs);

    // Validate group dimension compatibility (A[0] can be 1 for broadcasting or match shapeIndices[1]])
    if (shapeA[0] != 1 && shapeA[0] != shapeIndices[1]) {
        OPENVINO_ASSERT(false,
                        "GatherMatmul: incompatible group dimensions. ",
                        "A[0] = ",
                        shapeA[0],
                        " must be 1 or equal to the indices group = ",
                        shapeIndices[1]);
    }

    // Validate indices dimensions
    OPENVINO_ASSERT(shapeIndices[0] == M,
                    "GatherMatmul: indices first dimension must match A's seq_len. ",
                    "indices[0] = ",
                    shapeIndices[0],
                    ", A[1] = ",
                    M);

    // Compute output shape from matmul
    // matmul([M, K], [N, K].T) = [M, N]
    const size_t matmul_M = m_transpose_a ? K_A : M;
    const size_t matmul_N = m_transpose_b ? K : N;

    // Output shape: [group_size, M, N]
    const size_t group_size = shapeIndices[1];

    VectorDims outputShape = {group_size, matmul_M, matmul_N};

    return {{std::move(outputShape)}, ShapeInferStatus::success};
}

}  // namespace ov::intel_cpu::node
