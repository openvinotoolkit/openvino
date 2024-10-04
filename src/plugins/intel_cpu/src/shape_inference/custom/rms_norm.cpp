// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_norm.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class RMSNormShapeInfer : public ShapeInferEmptyPads {
public:
    RMSNormShapeInfer() {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& dims = input_shapes.front().get();
        return {{dims}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

ShapeInferPtr RMSNormShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<RMSNormShapeInfer>();
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
