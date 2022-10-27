// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference_cpu.hpp"
#include "shape_inference.hpp"
#include <ie_ngraph_utils.hpp>

namespace ov {
namespace intel_cpu {

class NgraphShapeInfer : public IShapeInfer {
public:
    NgraphShapeInfer(std::shared_ptr<IShapeInferCommon> shape_infer, IShapeInfer::port_mask_t port_mask) :
        m_shape_infer(shape_infer), m_port_mask(port_mask) {}

    std::vector<VectorDims> infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;

    // infer may generate padding as by-product, these APIs is designed to retrieve them back
    const ov::CoordinateDiff& get_pads_begin() override {
        return m_shape_infer->get_pads_begin();
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return m_shape_infer->get_pads_end();
    }
    port_mask_t getPortMask() const override {
        return m_port_mask;
    }
private:
    std::shared_ptr<IShapeInferCommon> m_shape_infer;
    IShapeInfer::port_mask_t m_port_mask;
};

std::vector<VectorDims> NgraphShapeInfer::infer(
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
    std::vector<StaticShape> output_shapes = m_shape_infer->infer(input_static_shapes, input_values);

    std::vector<VectorDims> result(output_shapes.size());
    std::transform(output_shapes.begin(), output_shapes.end(), result.begin(), [](const StaticShape& s) {
        return s.to_shape();
    });

    return result;
}

ShapeInferPtr DefaultShapeInferFactory::makeShapeInfer() const {
    return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), m_port_mask);
}

}   // namespace intel_cpu
}   // namespace ov