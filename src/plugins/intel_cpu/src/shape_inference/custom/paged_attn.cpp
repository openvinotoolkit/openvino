// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn.hpp"

#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class PAShapeInfer : public ShapeInferEmptyPads {
public:
    PAShapeInfer() {}

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& query_dims = input_shapes.front().get();

        return {{query_dims}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

ShapeInferPtr PAShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<PAShapeInfer>();
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
