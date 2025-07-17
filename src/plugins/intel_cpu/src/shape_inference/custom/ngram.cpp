// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/ngram.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "ngram.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"

namespace ov::intel_cpu::node {

Result NgramShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    auto output_shape = input_shapes[0].get();
    output_shape[1] *= m_k;
    return {{std::move(output_shape)}, ShapeInferStatus::success};
}

ShapeInferPtr NgramShapeInferFactory::makeShapeInfer() const {
    auto ngram = ov::as_type_ptr<NgramNode>(m_op);
    OPENVINO_ASSERT(ngram, "Wrong operation type");
    return std::make_shared<NgramShapeInfer>(ngram->get_k());
}
}  // namespace ov::intel_cpu::node
