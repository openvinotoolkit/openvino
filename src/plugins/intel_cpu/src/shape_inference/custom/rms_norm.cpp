// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_norm.hpp"

namespace ov::intel_cpu::node {

class RMSNormShapeInfer : public ShapeInferEmptyPads {
public:
    RMSNormShapeInfer() = default;

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& dims = input_shapes.front().get();
        return {{dims}, ShapeInferStatus::success};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

ShapeInferPtr RMSNormShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<RMSNormShapeInfer>();
}

}  // namespace ov::intel_cpu::node
