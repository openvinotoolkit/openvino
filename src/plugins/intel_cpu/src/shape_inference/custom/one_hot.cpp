// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/one_hot.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"

namespace ov::intel_cpu::node {

/**
 * Implements One Hot shape inference algorithm. The output shape is the input `indices` tensor shape, where a new axis
 * of size `depth` is inserted at the dimension defined by the `axis` parameter.
 *
 */
Result OneHotShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                               const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    auto depth = data_dependency.at(1)->getDataAs<int32_t>()[0];
    OPENVINO_ASSERT(depth >= 0, "OneHot depth value can't be negative.");
    auto result = input_shapes.front().get();
    auto depth_pos = result.begin();
    if (!result.empty()) {
        depth_pos += m_axis;
    }
    result.insert(depth_pos, depth);

    return {{std::move(result)}, ShapeInferStatus::success};
}

ShapeInferPtr OneHotShapeInferFactory::makeShapeInfer() const {
    auto oneHot = ov::as_type_ptr<const ov::op::v1::OneHot>(m_op);
    OPENVINO_ASSERT(oneHot, "Unexpected op type in OneHot shape inference factory: ", m_op->get_type_name());
    auto axis = oneHot->get_axis();
    auto dstShape = oneHot->get_output_partial_shape(0);
    int output_dims_size = dstShape.size();
    if (0 == output_dims_size) {
        output_dims_size = 1;
    }
    if (axis < 0) {
        axis += output_dims_size;
    }
    return std::make_shared<OneHotShapeInfer>(axis);
}
}  // namespace ov::intel_cpu::node
