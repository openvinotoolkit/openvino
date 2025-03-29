// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.hpp"

#include "utils.hpp"

namespace ov::intel_cpu::node {

Result FCShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                           const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const VectorDims& activationShape = input_shapes[0].get();
    const VectorDims& weightShape = input_shapes[1].get();
    size_t activationRank = activationShape.size();
    size_t channelRank = 1;

    // activation   weight    output_shape
    // NCHW         CoCHW     NCo
    // TNC          CoC       TNCo
    // NC           CoC       NCo
    VectorDims outputShape(out_rank, 1);
    // set Co
    outputShape.back() = std::accumulate(weightShape.begin(), weightShape.end() - 1, 1, std::multiplies<>());
    // set batch dims
    size_t batchRank = activationRank - channelRank;
    size_t startIdx = out_rank - batchRank - 1;
    for (size_t i = 0; i < batchRank; i++) {
        outputShape[i + startIdx] = activationShape[i];
    }

    return {{std::move(outputShape)}, ShapeInferStatus::success};
}
}  // namespace ov::intel_cpu::node
