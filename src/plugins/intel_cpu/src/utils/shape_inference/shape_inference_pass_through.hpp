// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shape_inference_cpu.hpp"

namespace ov {
namespace intel_cpu {

class ShapeInferPassThrough final : public IShapeInfer {
public:
    ShapeInferPassThrough() = default;
    std::vector<VectorDims> infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        IE_ASSERT(!input_shapes.empty());
        return {input_shapes.front()};
    }

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    const ov::CoordinateDiff& get_pads_begin() override {
        return empty_vec;
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return empty_vec;
    }
    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
private:
    ov::CoordinateDiff empty_vec{};
};

class PassTroughShapeInferFactory final : public ShapeInferFactory {
public:
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<ShapeInferPassThrough>();
    }
};

} // namespace intel_cpu
} // namespace ov