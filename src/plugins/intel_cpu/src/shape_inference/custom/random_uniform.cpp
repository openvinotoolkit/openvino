// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"
#include <openvino/op/random_uniform.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

// TODO: remove after fixing the issue 123011
IShapeInfer::Result RandomUniformShapeInfer::infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    VectorDims dims;
    const auto& mem = data_dependency.at(0);
    const auto rank = mem->getShape().getElementsCount();
    auto shape_prc = mem->getDesc().getPrecision();
    switch (shape_prc) {
        case InferenceEngine::Precision::I32: {
            auto data = reinterpret_cast<const int32_t*>(mem->getData());
            dims.assign(data, data + rank);
        } break;
        case InferenceEngine::Precision::I64: {
            auto data = reinterpret_cast<const int64_t*>(mem->getData());
            dims.assign(data, data + rank);
        } break;
        default:
            OPENVINO_THROW("Unexpected Shape input precision: ", shape_prc);
    }

    return {{dims}, ShapeInferStatus::success};
}

RandomUniformShapeInferFactory::RandomUniformShapeInferFactory(const std::shared_ptr<ov::Node>& op) : m_op(op) {
    OPENVINO_ASSERT(ov::is_type<const op::v8::RandomUniform>(m_op),
            "Unexpected op type in RandomUniform shape inference factory: ", m_op->get_type_name());
}

ShapeInferPtr RandomUniformShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<RandomUniformShapeInfer>();
}

} // namespace node
} // namespace intel_cpu
} // namespace ov
