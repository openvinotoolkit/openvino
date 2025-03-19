// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"

#include "utils.hpp"

namespace ov::intel_cpu::node {

TransposeShapeInfer::TransposeShapeInfer(const size_t& out_rank, const std::vector<size_t>& axes_vec)
    : m_out_rank(out_rank),
      m_axes_vec(axes_vec),
      m_outputShape(out_rank, 1),
      m_needReverse(axes_vec.empty()) {}

Result TransposeShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                  const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const VectorDims& shapeIn = input_shapes[0].get();
    if (m_needReverse) {
        for (size_t i = 0; i < m_out_rank; ++i) {
            m_outputShape[i] = shapeIn[m_out_rank - 1 - i];
        }
    } else {
        for (size_t i = 0; i < m_out_rank; ++i) {
            m_outputShape[i] = shapeIn[m_axes_vec[i]];
        }
    }
    return {{m_outputShape}, ShapeInferStatus::success};
}
ShapeInferPtr TransposeShapeInferFactory::makeShapeInfer() const {
    if (const auto order = ov::as_type_ptr<const ov::op::v0::Constant>(
            m_op->get_input_node_shared_ptr(ov::op::v1::Transpose::ORDER))) {
        const auto axes_vec = order->cast_vector<size_t>();
        return std::make_shared<TransposeShapeInfer>(m_op->get_output_partial_shape(0).rank().get_length(), axes_vec);
    }
    return std::make_shared<TransposeDynShapeInfer>();
}
}  // namespace ov::intel_cpu::node
