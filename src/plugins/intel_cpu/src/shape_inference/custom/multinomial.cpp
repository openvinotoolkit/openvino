// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include <openvino/op/multinomial.hpp>

#include "multinomial_shape_inference.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/static_shape.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
using namespace InferenceEngine;
Result MultinomialShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                    const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& input_shape = input_shapes[0].get();
    const auto& num_samples_memory = data_dependency.at(1);
    size_t num_samples;
    if (num_samples_memory->getDesc().getPrecision() == InferenceEngine::Precision::I32) {
        num_samples = reinterpret_cast<const int*>(num_samples_memory->getData())[0];
    } else {  // I64
        num_samples = reinterpret_cast<const int64_t*>(num_samples_memory->getData())[0];
    }
    return {{{input_shape[0], num_samples}}, ShapeInferStatus::success};
}

ShapeInferPtr MultinomialShapeInferFactory::makeShapeInfer() const {
    if (const auto multinomial = ov::as_type_ptr<const ov::op::v13::Multinomial>(m_op)) {
        return std::make_shared<MultinomialShapeInfer>();
    } else {
        OPENVINO_THROW("Unexpected operation type in the Multinomial shape inference factory");
    }
}
}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
