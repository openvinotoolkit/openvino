// Copyright (C) 2018-2025 Intel Corporation
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
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"

namespace ov::intel_cpu::node {

/**
 * Implements Color Convert shape inference algorithm. Depending on wether it has only single plain H dimension is
 * passed through or recalculated as 2/3 of the initial size.
 *
 */
Result ColorConvertShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                     [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& dims = input_shapes.front().get();
    OPENVINO_ASSERT(dims.size() == 4, "NV12Converter node has incorrect input dimensions");
    return {m_singlePlain ? std::vector<VectorDims>{{dims[ColorConvert::Converter::N_DIM],
                                                     dims[ColorConvert::Converter::H_DIM] * 2 / 3,
                                                     dims[ColorConvert::Converter::W_DIM],
                                                     3}}
                          : std::vector<VectorDims>{{dims[ColorConvert::Converter::N_DIM],
                                                     dims[ColorConvert::Converter::H_DIM],
                                                     dims[ColorConvert::Converter::W_DIM],
                                                     3}},
            ShapeInferStatus::success};
}

ShapeInferPtr ColorConvertShapeInferFactory::makeShapeInfer() const {
    bool isSinglePlain = m_op->get_input_size() == 1;
    return std::make_shared<ColorConvertShapeInfer>(isSinglePlain);
}
}  // namespace ov::intel_cpu::node
