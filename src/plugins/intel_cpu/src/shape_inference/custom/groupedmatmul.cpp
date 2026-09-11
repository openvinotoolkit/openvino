// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "groupedmatmul.hpp"

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

Result GroupedMatMulShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                      [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    OPENVINO_DEBUG_ASSERT(input_shapes.size() >= 2,
                          "GroupedMatMul shape inference expects at least 2 inputs, got: ",
                          input_shapes.size());

    const VectorDims& shapeA = input_shapes[0].get();
    const VectorDims& shapeB = input_shapes[1].get();

    OPENVINO_DEBUG_ASSERT(any_of(shapeA.size(), 2U, 3U),
                          "GroupedMatMul mat_a must be 2D or 3D, got shape: ",
                          vec2str(shapeA));
    OPENVINO_DEBUG_ASSERT(shapeB.size() == 3, "GroupedMatMul mat_b must be 3D, got shape: ", vec2str(shapeB));

    // mat_b is stored pre-transposed as [G, N, K]
    const size_t N = shapeB[1];
    const size_t K = shapeB[2];

    OPENVINO_ASSERT(shapeA.back() == K,
                    "GroupedMatMul: K dimension mismatch. mat_a shape: ",
                    vec2str(shapeA),
                    ", mat_b shape: ",
                    vec2str(shapeB));

    VectorDims outputShape;
    if (shapeA.size() == 2) {
        // 2D x 3D: rows of mat_a are partitioned among the groups by offsets
        outputShape = {shapeA[0], N};
    } else {
        // 3D x 3D: batched uniform groups
        OPENVINO_ASSERT(shapeA[0] == shapeB[0],
                        "GroupedMatMul: group dimension mismatch. mat_a shape: ",
                        vec2str(shapeA),
                        ", mat_b shape: ",
                        vec2str(shapeB));
        outputShape = {shapeA[0], shapeA[1], N};
    }

    return {{std::move(outputShape)}, ShapeInferStatus::success};
}

}  // namespace ov::intel_cpu::node
