// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot.hpp"

#include "openvino/opsets/opset1.hpp"
#include "utils.hpp"

namespace ov::intel_cpu::node {

/**
 * Implements One Hot shape inference algorithm. The output shape is the input `indices` tensor shape, where a new axis
 * of size `depth` is inserted at the dimension defined by the `axis` parameter.
 *
 */
Result OneHotShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                               const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    auto depth = data_dependency.at(1)->getDataAs<int32_t>()[0];
    if (depth < 0) {
        OPENVINO_THROW("OneHot depth value can't be negative.");
    }
    auto result = input_shapes.front().get();
    result.insert(result.begin() + m_axis, depth);

    return {{std::move(result)}, ShapeInferStatus::success};
}

ShapeInferPtr OneHotShapeInferFactory::makeShapeInfer() const {
    auto oneHot = ov::as_type_ptr<const ov::opset1::OneHot>(m_op);
    if (!oneHot) {
        OPENVINO_THROW("Unexpected op type in OneHot shape inference factory: ", m_op->get_type_name());
    }
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
