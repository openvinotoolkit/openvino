// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice.hpp"
#include "utils.hpp"
#include "ie_ngraph_utils.hpp"
#include "slice_shape_inference.hpp"
#include <shape_inference/shape_inference_ngraph.hpp>

namespace ov {
namespace intel_cpu {
namespace node {
using namespace InferenceEngine;
StridedSliceShapeInfer::StridedSliceShapeInfer(size_t output_size,
        std::unordered_set<int64_t> begin_mask,
        std::unordered_set<int64_t> end_mask,
        std::unordered_set<int64_t> new_axis_mask,
        std::unordered_set<int64_t> shrink_axis_mask)
    : m_outputShape(output_size, 1),
    m_begin_mask_set(std::move(begin_mask)),
    m_end_mask_set(std::move(end_mask)),
    m_new_axis_mask_set(std::move(new_axis_mask)),
    m_shrink_axis_mask_set(std::move(shrink_axis_mask)) {}

Result StridedSliceShapeInfer::infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    // align with intel_cpu::node::StridedSlice
    static constexpr size_t DATA_ID = 0, BEGIN_ID = 1, END_ID = 2, STRIDE_ID = 3;
    const VectorDims& shapeIn = input_shapes[DATA_ID].get();
    const VectorDims& shapeBegin = input_shapes[BEGIN_ID].get();
    if (data_dependency.at(BEGIN_ID)->getDesc().getPrecision() != ov::element::i32 ||
            data_dependency.at(END_ID)->getDesc().getPrecision() != ov::element::i32 ||
            data_dependency.at(STRIDE_ID)->getDesc().getPrecision() != ov::element::i32) {
        OPENVINO_THROW("The data type of begin/end/stride is NOT I32, which is unexpected!");
    }
    auto beginPtr = reinterpret_cast<int32_t *>(data_dependency.at(BEGIN_ID)->getData());
    auto endPtr = reinterpret_cast<int32_t *>(data_dependency.at(END_ID)->getData());
    auto stridePtr = reinterpret_cast<int32_t *>(data_dependency.at(STRIDE_ID)->getData());

    for (size_t i = 0, new_idx = 0; i < shapeIn.size(); ++i) {
        if (m_new_axis_mask_set.count(i)) {
            // deal with new_axis_mask
            m_outputShape[new_idx] = 1;
            m_outputShape[new_idx+1] = shapeIn[i];
            new_idx+=2;
        } else if (!m_shrink_axis_mask_set.count(i)) {
            // deal with begin_mask and end_mask
            if ((i >= shapeBegin[0]) || (shapeIn[i] == 0)) {
                m_outputShape[new_idx] = shapeIn[i];
            } else {
                int32_t begin = 0;
                int32_t end = 0;
                if (stridePtr[i] < 0) {
                    begin = m_begin_mask_set.count(i) ? shapeIn[i] : beginPtr[i];
                    end  = m_end_mask_set.count(i) ? (-1 - shapeIn[i]) : endPtr[i];
                } else {
                    begin = m_begin_mask_set.count(i) ? 0 : beginPtr[i];
                    end  = m_end_mask_set.count(i) ? shapeIn[i] : endPtr[i];
                }
                m_outputShape[new_idx] = ov::op::slice::get_sliced_value(shapeIn[i], begin, end, stridePtr[i]);
            }
            new_idx += 1;
        }
    }
    return {{m_outputShape}, ShapeInferStatus::success};
}

ShapeInferPtr StridedSliceShapeInferFactory::makeShapeInfer() const {
    if (const auto Slice_op = ov::as_type_ptr<const ov::op::v8::Slice>(m_op)) {
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), port_mask);
    } else if (const auto StridedSlice_op = ov::as_type_ptr<const ov::op::v1::StridedSlice>(m_op)) {
        const auto& ellipsis_mask = StridedSlice_op->get_ellipsis_mask();
        if (std::any_of(ellipsis_mask.begin(), ellipsis_mask.end(), [](int64_t x){ return x == 1; })) {
            return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), port_mask);
        } else {
            auto vec_to_set = [](const std::vector<int64_t>& vec){
                std::unordered_set<int64_t> to_set;
                for (size_t i = 0; i < vec.size(); ++i) {
                    if (vec[i] == 1) {
                        to_set.emplace(i);
                    }
                }
                return to_set;
            };
            return std::make_shared<StridedSliceShapeInfer>(
                    m_op->get_output_partial_shape(0).rank().get_length(),
                    vec_to_set(StridedSlice_op->get_begin_mask()),
                    vec_to_set(StridedSlice_op->get_end_mask()),
                    vec_to_set(StridedSlice_op->get_new_axis_mask()),
                    vec_to_set(StridedSlice_op->get_shrink_axis_mask()));
        }
    } else {
        OPENVINO_THROW("not Slice or StridedSlice");
    }
}

} // namespace node
} // namespace intel_cpu
} // namespace ov
