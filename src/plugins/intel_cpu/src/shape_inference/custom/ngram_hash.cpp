// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_hash.hpp"

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "openvino/op/ngram_hash_function.hpp"

namespace ov::intel_cpu::node {

IShapeInfer::Result NgramHashShapeInfer::infer(
    const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
    [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    // [..., ngram_size] -> [..., H]
    OPENVINO_ASSERT(input_shapes.size() == 1, "NGramHashFunction requires one input shape.");
    auto output_shape = input_shapes[0].get();
    OPENVINO_ASSERT(!output_shape.empty(), "NGramHashFunction input must have rank >= 1.");
    OPENVINO_ASSERT(output_shape.back() == m_ngram_size,
                    "The last dimension of the ngrams input must be equal to ngram_size (",
                    m_ngram_size,
                    "). Got: ",
                    output_shape.back());
    output_shape.back() = m_heads;
    return {{std::move(output_shape)}, ShapeInferStatus::success};
}

ShapeInferPtr NgramHashShapeInferFactory::makeShapeInfer() const {
    auto ngram_hash = ov::as_type_ptr<ov::op::internal::NGramHashFunction>(m_op);
    OPENVINO_ASSERT(ngram_hash, "Wrong operation type");
    return std::make_shared<NgramHashShapeInfer>(ngram_hash->get_num_heads(),
                                                 static_cast<size_t>(ngram_hash->get_ngram_size()));
}

}  // namespace ov::intel_cpu::node
