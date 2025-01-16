// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice.hpp"
#include "utils.hpp"
#include "slice_shape_inference.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

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
    auto beginPtr = data_dependency.at(BEGIN_ID)->getDataAs<int32_t>();
    auto endPtr = data_dependency.at(END_ID)->getDataAs<int32_t>();
    auto stridePtr = data_dependency.at(STRIDE_ID)->getDataAs<int32_t>();

    const auto begin_size = shapeBegin[0];

    auto gen_new_sliced_value = [&](size_t cur_idx, size_t in_idx) -> size_t {
        if ((cur_idx >= begin_size) || (shapeIn[in_idx] == 0)) {
            return shapeIn[in_idx];
        } else {
            int32_t begin = 0, end = 0;
            if (stridePtr[cur_idx] < 0) {
                begin = m_begin_mask_set.count(cur_idx) ? shapeIn[in_idx] : beginPtr[cur_idx];
                end  = m_end_mask_set.count(cur_idx) ? (-1 - shapeIn[in_idx]) : endPtr[cur_idx];
            } else {
                begin = m_begin_mask_set.count(cur_idx) ? 0 : beginPtr[cur_idx];
                end  = m_end_mask_set.count(cur_idx) ? shapeIn[in_idx] : endPtr[cur_idx];
            }
            return ov::op::slice::get_sliced_value(shapeIn[in_idx], begin, end, stridePtr[cur_idx]);
        }
    };

    for (size_t in_idx = 0, out_idx = 0; in_idx < shapeIn.size(); ++in_idx) {
        if (m_new_axis_mask_set.count(in_idx)) {
            // deal with new_axis_mask
            m_outputShape[out_idx] = 1;
            out_idx++;
            // deal with current axis
            m_outputShape[out_idx] = gen_new_sliced_value(out_idx, in_idx);
            out_idx++;
        } else if (!m_shrink_axis_mask_set.count(in_idx)) {
            // deal with begin_mask and end_mask
            m_outputShape[out_idx] = gen_new_sliced_value(in_idx, in_idx);
            out_idx++;
        }
    }
    return {{m_outputShape}, ShapeInferStatus::success};
}

ShapeInferPtr StridedSliceShapeInferFactory::makeShapeInfer() const {
    if (const auto Slice_op = ov::as_type_ptr<const ov::op::v8::Slice>(m_op)) {
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), port_mask);
    } else if (const auto SliceScatter_op = ov::as_type_ptr<const ov::op::v15::SliceScatter>(m_op)) {
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), PortMask(2, 3, 4, 5));
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
