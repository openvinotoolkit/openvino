// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include "utils.hpp"

namespace ov::intel_cpu::node {

/**
 * Implements Eltwise shape inference algorithm. The algorithm is based on broadcasting all the input shapes
 * according to the NUMPY broadcast rule. This implementation is more lightweight than the ngraph one.
 *
 */
Result EltwiseShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    size_t max_rank = 0;
    size_t max_rank_idx = 0;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        auto item_rank = input_shapes[i].get().size();
        if (item_rank > max_rank) {
            max_rank = item_rank;
            max_rank_idx = i;
        }
    }
    auto output_shape = input_shapes[max_rank_idx].get();
    // use NUMPY broadcast rule
    for (size_t i = 0; i < input_shapes.size(); i++) {
        if (i == max_rank_idx) {
            continue;
        }

        auto& input_shape = input_shapes[i].get();
        if (input_shape.size() > output_shape.size()) {
            OPENVINO_THROW("Eltwise shape infer input and output shapes rank mismatch");
        }
        size_t offset = output_shape.size() - input_shape.size();
        for (size_t j = 0; j < input_shape.size(); ++j) {
            if (input_shape[j] != output_shape[offset + j]) {
                if (output_shape[offset + j] == 1) {
                    output_shape[offset + j] = input_shape[j];
                } else {
                    if (input_shape[j] != 1) {
                        OPENVINO_THROW("Eltwise shape infer input shapes dim index: ", j, " mismatch");
                    }
                }
            }
        }
    }
    return {{std::move(output_shape)}, ShapeInferStatus::success};
}

}  // namespace ov::intel_cpu::node
