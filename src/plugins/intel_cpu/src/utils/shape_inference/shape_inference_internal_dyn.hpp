// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference_cpu.hpp"

namespace ov {
namespace intel_cpu {

class InternalDynShapeInfer final : public ShapeInferEmptyPads {
public:
    InternalDynShapeInfer() = default;
    std::vector<VectorDims> infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        IE_ASSERT(false) << "InternalDynShapeInfer infer method unexpected call";
        return {};
    }

    port_mask_t get_port_mask() const override {
        return FULL_PORT_MASK;
    }
};

class InternalDynShapeInferFactory final : public ShapeInferFactory {
public:
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<InternalDynShapeInfer>();
    }
};

} // namespace intel_cpu
} // namespace ov