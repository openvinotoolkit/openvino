// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "priorbox_clustered.hpp"

#include "openvino/opsets/opset1.hpp"
#include "utils.hpp"

namespace ov::intel_cpu::node {

/**
 * Implements Prior Box Clustered shape inference algorithm. The output shape is [2,  4 * height * width *
 * number_of_priors]. `number_of_priors` is an attribute of the operation. heigh and width are in the the first input
 * parameter.
 *
 */
Result PriorBoxClusteredShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                          const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto* in_data = data_dependency.at(0)->getDataAs<const int>();
    const int H = in_data[0];
    const int W = in_data[1];
    const auto output = static_cast<size_t>(4 * H * W * m_number_of_priors);
    return {{{2, output}}, ShapeInferStatus::success};
}

ShapeInferPtr PriorBoxClusteredShapeInferFactory::makeShapeInfer() const {
    auto priorBox = ov::as_type_ptr<const ov::opset1::PriorBoxClustered>(m_op);
    if (!priorBox) {
        OPENVINO_THROW("Unexpected op type in PriorBoxClustered shape inference factory: ", m_op->get_type_name());
    }
    const auto& attrs = priorBox->get_attrs();
    auto number_of_priors = attrs.widths.size();
    return std::make_shared<PriorBoxClusteredShapeInfer>(number_of_priors);
}

}  // namespace ov::intel_cpu::node
