// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference_ngraph.hpp"
#include <memory>
#include "memory_accessor.hpp"

using namespace ov::intel_cpu;

IShapeInfer::Result
NgraphShapeInfer::infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const auto& iranks = m_shape_infer->get_input_ranks();
    IE_ASSERT(iranks.size() <= input_shapes.size()) << "Too few input shapes passed to Shape infer.";
    std::vector<StaticShapeRef> input_static_shapes;
    std::map<size_t, ov::HostTensorPtr> input_values;

    input_static_shapes.reserve(input_shapes.size());
    IShapeInferCommon::Result shape_infer_result;

    for (size_t port = 0; port < iranks.size(); port++) {
        if (iranks[port] == 0) {
            input_static_shapes.emplace_back();
        } else {
            input_static_shapes.emplace_back(input_shapes[port].get());
        }
    }

    if (m_has_implemented_accessor) {
        auto static_shape_infer = std::static_pointer_cast<IStaticShapeInfer>(m_shape_infer);
        // call shape inference API
        shape_infer_result = static_shape_infer->infer(input_static_shapes, MemoryAccessor(data_dependency, iranks));
    } else {
        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> input_values;
        for (size_t port = 0; port < iranks.size(); port++) {
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
                    memPtr->getData());
            }
        }
        // call shape inference API
        shape_infer_result = m_shape_infer->infer(input_static_shapes, input_values);
    }
    // call shape inference API
    auto shape_infer_result = m_shape_infer->infer(input_static_shapes, ov::make_tensor_accessor(input_values));

    Result result{{}, shape_infer_result ? ShapeInferStatus::success : ShapeInferStatus::skip};

    if (shape_infer_result) {
        result.dims.reserve(shape_infer_result->size());
        std::transform(shape_infer_result->begin(),
                       shape_infer_result->end(),
                       std::back_inserter(result.dims),
                       [](StaticShape& s) {
                           return std::move(*s);
                       });
    }

    return result;
}
