// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/color_convert.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "color_convert.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/bgr_to_nv12.hpp"
#include "openvino/op/rgb_to_nv12.hpp"
#include "openvino/op/util/convert_color_to_nv12_base.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"

namespace ov::intel_cpu::node {

/**
 * Implements Color Convert shape inference algorithm. Depending on whether it has only single plain H dimension is
 * passed through or recalculated as 2/3 of the initial size.
 *
 */
Result ColorConvertShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                     [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& dims = input_shapes.front().get();
    OPENVINO_ASSERT(dims.size() == 4, "NV12Converter node has incorrect input dimensions");
    return {m_singlePlane ? std::vector<VectorDims>{{dims[ColorConvert::Converter::N_DIM],
                                                     dims[ColorConvert::Converter::H_DIM] * 2 / 3,
                                                     dims[ColorConvert::Converter::W_DIM],
                                                     3}}
                          : std::vector<VectorDims>{{dims[ColorConvert::Converter::N_DIM],
                                                     dims[ColorConvert::Converter::H_DIM],
                                                     dims[ColorConvert::Converter::W_DIM],
                                                     3}},
            ShapeInferStatus::success};
}

Result ColorConvertToNV12ShapeInfer::infer(
    const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
    [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& dims = input_shapes.front().get();
    OPENVINO_ASSERT(dims.size() == 4, "RGBtoNV12/BGRtoNV12 node expects 4 input dimensions");

    if (m_singlePlane) {
        // Single-plane output: [N, H*3/2, W, 1]
        return {{VectorDims{dims[ColorConvert::Converter::N_DIM],
                            dims[ColorConvert::Converter::H_DIM] * 3 / 2,
                            dims[ColorConvert::Converter::W_DIM],
                            1}},
                ShapeInferStatus::success};
    }
    // Two-plane output: Y=[N, H, W, 1]  UV=[N, H/2, W/2, 2]
    return {{VectorDims{dims[ColorConvert::Converter::N_DIM],
                        dims[ColorConvert::Converter::H_DIM],
                        dims[ColorConvert::Converter::W_DIM],
                        1},
             VectorDims{dims[ColorConvert::Converter::N_DIM],
                        dims[ColorConvert::Converter::H_DIM] / 2,
                        dims[ColorConvert::Converter::W_DIM] / 2,
                        2}},
            ShapeInferStatus::success};
}

ShapeInferPtr ColorConvertShapeInferFactory::makeShapeInfer() const {
    if (ov::is_type<ov::op::v17::RGBtoNV12>(m_op) || ov::is_type<ov::op::v17::BGRtoNV12>(m_op)) {
        const auto base = ov::as_type_ptr<ov::op::util::ConvertColorToNV12Base>(m_op);
        const bool isSinglePlane = base && base->is_single_plane();
        return std::make_shared<ColorConvertToNV12ShapeInfer>(isSinglePlane);
    }
    bool isSinglePlane = m_op->get_input_size() == 1;
    return std::make_shared<ColorConvertShapeInfer>(isSinglePlane);
}
}  // namespace ov::intel_cpu::node
