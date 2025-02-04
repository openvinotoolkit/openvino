// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adaptive_pooling.hpp"

#include "utils.hpp"

namespace ov::intel_cpu::node {

/**
 * Implements Adaptive Pooling shape inference algorithm. The output tensor shape consists of the input [N, C]
 * dimensions and the [D_out, H_out, W_out] dimensions, which are placed in the second input parameter.
 *
 */
Result AdaptivePoolingShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                        const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& inputDims = input_shapes[0].get();
    const auto& spatialDims = input_shapes[1].get();
    const auto inputRank = inputDims.size();
    const auto spatialDimsSize = spatialDims[0];

    VectorDims outputDims(inputRank);
    outputDims[0] = inputDims[0];
    outputDims[1] = inputDims[1];
    auto newSpatialDimsPtr = data_dependency.at(1)->getDataAs<int32_t>();
    for (size_t i = 0; i < spatialDimsSize; i++) {
        outputDims[i + 2] = newSpatialDimsPtr[i];
    }

    std::vector<VectorDims> result(m_outputs_count, outputDims);
    return {std::move(result), ShapeInferStatus::success};
}

ShapeInferPtr AdaptivePoolingShapeInferFactory::makeShapeInfer() const {
    size_t outputs_count = m_op->get_output_size();
    return std::make_shared<AdaptivePoolingShapeInfer>(outputs_count);
}

}  // namespace ov::intel_cpu::node
