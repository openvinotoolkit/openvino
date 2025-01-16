// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov {
namespace intel_cpu {
std::vector<StaticShapeRef> make_static_shape_refs(const ShapeVector& shapes) {
    std::vector<StaticShapeRef> out;
    out.reserve(shapes.size());
    for (auto& s : shapes) {
        out.emplace_back(s);
    }
    return out;
}

ShapeVector shape_inference(ov::Node* op,
                            const ShapeVector& input_shapes,
                            const std::unordered_map<size_t, Tensor>& constant_data) {
    const auto in_shapes = intel_cpu::make_static_shape_refs(input_shapes);
    const auto shape_infer = intel_cpu::make_shape_inference(op->shared_from_this());
    auto result = shape_infer->infer(in_shapes, make_tensor_accessor(constant_data));
    OPENVINO_ASSERT(result, "There are no output shapes in shape inference result");
    return *result;
}
}  // namespace intel_cpu
}  // namespace ov
