// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice.hpp"

#include "openvino/core/type.hpp"
#include "shape_inference/shape_inference.hpp"
#include "slice_shape_inference.hpp"
#include "utils.hpp"

namespace ov::intel_cpu::node {

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

Result StridedSliceShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
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
        }
        int32_t begin = 0, end = 0;
        if (stridePtr[cur_idx] < 0) {
            begin = m_begin_mask_set.count(cur_idx) ? shapeIn[in_idx] : beginPtr[cur_idx];
            end = m_end_mask_set.count(cur_idx) ? (-1 - shapeIn[in_idx]) : endPtr[cur_idx];
        } else {
            begin = m_begin_mask_set.count(cur_idx) ? 0 : beginPtr[cur_idx];
            end = m_end_mask_set.count(cur_idx) ? shapeIn[in_idx] : endPtr[cur_idx];
        }
        return ov::op::slice::get_sliced_value(shapeIn[in_idx], begin, end, stridePtr[cur_idx]);
    };

    const auto shapeInSize = shapeIn.size();
    const auto newAxisMaskSize = m_new_axis_mask_set.size();
    const auto maxAxisSize = shapeInSize + newAxisMaskSize;
    const auto outputShapeSize = m_outputShape.size();
    bool newAxis = false;
    bool shrinkAxis = false;
    // because has already initialized the elements of m_outputShape as value 1,
    // so don't need m_outputShape[out_idx] = 1 when newAxis,
    // so also don't need to check if there are new axis at the end of ShapeIn.
    for (size_t axis_idx = 0, out_idx = 0, in_idx = 0;
         axis_idx < maxAxisSize && in_idx < shapeInSize && out_idx < outputShapeSize;
         axis_idx++) {
        newAxis = m_new_axis_mask_set.count(axis_idx);
        shrinkAxis = m_shrink_axis_mask_set.count(axis_idx);
        if (newAxis) {
            // from test when shrinkAxis && newAxis, only newAxis is working in NgraphShapeInfer,
            // so merge if(newAxis) and if(shrinkAxis && newAxis) together.
            out_idx++;
        } else if (shrinkAxis) {
            in_idx++;
        } else {
            m_outputShape[out_idx] = gen_new_sliced_value(axis_idx, in_idx);
            in_idx++;
            out_idx++;
        }
    }

    return {{m_outputShape}, ShapeInferStatus::success};
}

ShapeInferPtr StridedSliceShapeInferFactory::makeShapeInfer() const {
    if (ov::is_type_any_of<const ov::op::v8::Slice, const ov::op::v15::SliceScatter>(m_op)) {
        return make_shape_inference(m_op);
    }
    if (const auto StridedSlice_op = ov::as_type_ptr<const ov::op::v1::StridedSlice>(m_op)) {
        const auto& ellipsis_mask = StridedSlice_op->get_ellipsis_mask();
        if (std::any_of(ellipsis_mask.begin(), ellipsis_mask.end(), [](int64_t x) {
                return x == 1;
            })) {
            return make_shape_inference(m_op);
        }
        auto vec_to_set = [](const std::vector<int64_t>& vec) {
            std::unordered_set<int64_t> to_set;
            for (size_t i = 0; i < vec.size(); ++i) {
                if (vec[i] == 1) {
                    to_set.emplace(i);
                }
            }
            return to_set;
        };
        return std::make_shared<StridedSliceShapeInfer>(m_op->get_output_partial_shape(0).rank().get_length(),
                                                        vec_to_set(StridedSlice_op->get_begin_mask()),
                                                        vec_to_set(StridedSlice_op->get_end_mask()),
                                                        vec_to_set(StridedSlice_op->get_new_axis_mask()),
                                                        vec_to_set(StridedSlice_op->get_shrink_axis_mask()));
    }
    OPENVINO_THROW("not Slice or StridedSlice");
}

}  // namespace ov::intel_cpu::node
