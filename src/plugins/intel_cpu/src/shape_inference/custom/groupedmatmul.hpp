// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

using Result = IShapeInfer::Result;

// GroupedMatMul-17 has two forms, both fully determined by the input shapes:
//   2D x 3D: A[T, K]    x B[G, N, K] (+ offsets[G]) -> [T, N]
//   3D x 3D: A[G, M, K] x B[G, N, K]                -> [G, M, N]
// The offsets tensor only redistributes rows of A among the groups, so its data is not needed here.
class GroupedMatMulShapeInfer : public ShapeInferEmptyPads {
public:
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

// GroupedMatMulShapeInfer is stateless — it derives everything from the input shapes — so unlike the
// other factories in this directory this one does not need to capture the op.
class GroupedMatMulShapeInferFactory : public ShapeInferFactory {
public:
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<GroupedMatMulShapeInfer>();
    }
};

}  // namespace ov::intel_cpu::node
