// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot.hpp"
#include "utils.hpp"
#include "ie_ngraph_utils.hpp"
#include <openvino/opsets/opset1.hpp>

namespace ov {
namespace intel_cpu {
namespace node {
using namespace InferenceEngine;

/**
 * Implements One Hot shape inference algorithm. The output shape is the input `indices` tensor shape, where a new axis
 * of size `depth` is inserted at the dimension defined by the `axis` parameter.
 *
 */
Result OneHotShapeInfer::infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    auto depth = reinterpret_cast<int32_t *>(data_dependency.at(1)->getData())[0];
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
    if (0 == output_dims_size) output_dims_size = 1;
    if (axis < 0) {
        axis += output_dims_size;
    }
    return std::make_shared<OneHotShapeInfer>(axis);
}
} // namespace node
} // namespace intel_cpu
} // namespace ov
