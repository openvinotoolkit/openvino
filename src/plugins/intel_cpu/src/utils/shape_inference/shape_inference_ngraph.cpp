// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference_ngraph.hpp"

using namespace ov::intel_cpu;

IShapeInfer::Result
NgraphShapeInfer::infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& iranks = m_shape_infer->get_input_ranks();
    IE_ASSERT(iranks.size() <= input_shapes.size()) << "Too few input shapes passed to Shape infer.";
    std::vector<StaticShape> input_static_shapes;
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> input_values;

    input_static_shapes.reserve(input_shapes.size());
    for (size_t port = 0; port < iranks.size(); port++) {
        if (iranks[port] == 0) {
            input_static_shapes.emplace_back();
        } else {
            input_static_shapes.emplace_back(input_shapes[port].get());
        }
        auto itr = data_dependency.find(port);
        if (itr != data_dependency.end()) {
            const auto& memPtr = itr->second;

            ov::Shape shape;

            // use scalar shape {} instead of {1} if required by shapeInference
            if (iranks[port] != 0) {
                shape = ov::Shape(memPtr->getStaticDims());
            }

            input_values[port] = std::make_shared<ngraph::runtime::HostTensor>(
                InferenceEngine::details::convertPrecision(memPtr->getDesc().getPrecision()),
                shape,
                memPtr->GetPtr());
        }
    }
    // call shape inference API
    auto shape_infer_result = m_shape_infer->infer(input_static_shapes, input_values);

    std::vector<VectorDims> result;
    result.reserve(shape_infer_result.shapes.size());
    for (const auto& shape : shape_infer_result.shapes) {
        result.emplace_back(shape.to_shape());
    }

    return {std::move(result), shape_infer_result.status};
}
