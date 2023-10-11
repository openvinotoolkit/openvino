// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/color_convert.h"
#include "color_convert.hpp"
#include "utils.hpp"
#include "ie_ngraph_utils.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
using namespace InferenceEngine;
/**
 * Implements Color Convert shape inference algorithm. Depending on wether it has only single plain H dimension is
 * passed through or recalculated as 2/3 of the initial size.
 *
 */
Result ColorConvertShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                     const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& dims = input_shapes.front().get();
    if (dims.size() != 4) {
        OPENVINO_THROW("NV12Converter node has incorrect input dimensions");
    }
    return { m_singlePlain
    ? std::vector<VectorDims>{ { dims[ColorConvert::Converter::N_DIM], dims[ColorConvert::Converter::H_DIM] * 2 / 3, dims[ColorConvert::Converter::W_DIM], 3 } }
    :
    std::vector<VectorDims>{ { dims[ColorConvert::Converter::N_DIM], dims[ColorConvert::Converter::H_DIM], dims[ColorConvert::Converter::W_DIM], 3 } },
    ShapeInferStatus::success };
}

ShapeInferPtr ColorConvertShapeInferFactory::makeShapeInfer() const {
    bool isSinglePlain = m_op->get_input_size() == 1;
    return std::make_shared<ColorConvertShapeInfer>(isSinglePlain);
}
} // namespace node
} // namespace intel_cpu
} // namespace ov
