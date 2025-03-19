// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/op/ngram.hpp"

#include "ngram.hpp"
#include "utils.hpp"

namespace ov::intel_cpu::node {

Result NgramShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    auto output_shape = input_shapes[0].get();
    output_shape[1] *= m_k;
    return {{std::move(output_shape)}, ShapeInferStatus::success};
}

ShapeInferPtr NgramShapeInferFactory::makeShapeInfer() const {
    auto ngram = ov::as_type_ptr<NgramNode>(m_op);
    if (!ngram) {
        OPENVINO_THROW("Wrong operation type");
    }
    return std::make_shared<NgramShapeInfer>(ngram->get_k());
}
}  // namespace ov::intel_cpu::node
