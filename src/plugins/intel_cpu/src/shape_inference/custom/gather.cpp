// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather.hpp"

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
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"

namespace ov::intel_cpu::node {

Result GatherShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                               const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;
    const auto& input_shape = input_shapes[GATHER_DATA].get();
    // Use VectorDims{} instead of {1} for Scalar
    const auto& indices_shape = m_isIndicesScalar ? VectorDims{} : input_shapes[GATHER_INDICES].get();
    if (!m_isAxisInputConst) {
        if (data_dependency.at(GATHER_AXIS)->getDesc().getPrecision() != ov::element::i32) {
            OPENVINO_THROW("Unsupported precision ",
                           data_dependency.at(GATHER_AXIS)->getDesc().getPrecision(),
                           " for axis tensor.");
        }
        m_axis = data_dependency.at(GATHER_AXIS)->getDataAs<const int32_t>()[0];
    }
    if (m_axis < 0) {
        m_axis += input_shape.size();
    }
    if (m_batchDims < 0) {
        m_batchDims += indices_shape.size();
    }
    VectorDims output_shape;
    output_shape.reserve(input_shape.size() + indices_shape.size() - m_batchDims - 1);
    output_shape.insert(output_shape.end(), input_shape.begin(), input_shape.begin() + m_axis);
    output_shape.insert(output_shape.end(), indices_shape.begin() + m_batchDims, indices_shape.end());
    output_shape.insert(output_shape.end(), input_shape.begin() + m_axis + 1, input_shape.end());
    return {{std::move(output_shape)}, ShapeInferStatus::success};
}

ShapeInferPtr GatherShapeInferFactory::makeShapeInfer() const {
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;
    bool isAxisInputConst = ov::is_type<ov::op::v0::Constant>(m_op->get_input_node_ptr(GATHER_AXIS));
    const auto& indicesShape = m_op->get_input_partial_shape(GATHER_INDICES);
    OPENVINO_ASSERT(indicesShape.rank().is_static(), "indicesShape do not support dynamic rank.");
    bool isIndicesScalar = indicesShape.rank().get_length() == 0;
    int axis = isAxisInputConst
                   ? ov::as_type<ov::op::v0::Constant>(m_op->get_input_node_ptr(GATHER_AXIS))->cast_vector<int>()[0]
                   : 0;
    int batchDims = 0;
    if (ov::is_type<ov::op::v8::Gather>(m_op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v8::Gather>(m_op)->get_batch_dims());
    } else if (ov::is_type<ov::op::v7::Gather>(m_op)) {
        batchDims = static_cast<int>(ov::as_type_ptr<ov::op::v7::Gather>(m_op)->get_batch_dims());
    }
    return std::make_shared<GatherShapeInfer>(isAxisInputConst, isIndicesScalar, axis, batchDims);
}

}  // namespace ov::intel_cpu::node
