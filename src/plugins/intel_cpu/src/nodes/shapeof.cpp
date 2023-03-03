// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shapeof.h"
#include <ngraph/opsets/opset1.hpp>
#include <utils/shape_inference/shape_inference_cpu.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
/**
 * Implements Shape Of shape inference algorithm. The output shape is simply a 1D tensor with the size of the input tensor
 * rank.
 *  
 */
class ShapeOfShapeInfer : public ShapeInferEmptyPads {
public:
    ShapeOfShapeInfer() = default;
    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        IE_ASSERT(!input_shapes.empty());
        return {{VectorDims{input_shapes.front().get().size()}}, ShapeInferStatus::success};
    }

    port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }
};

class ShapeOfShapeInferFactory : public ShapeInferFactory {
public:
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<ShapeOfShapeInfer>();
    }
};
} // namespace

bool ShapeOf::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ngraph::op::v0::ShapeOf::get_type_info_static(),
                    ngraph::op::v3::ShapeOf::get_type_info_static())) {
            errorMessage = "Node is not an instance of ShapeOf form the operation set v1 or v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ShapeOf::ShapeOf(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, ShapeOfShapeInferFactory()) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "ShapeOf layer with name '" + getName() + "' ";
        if (op->get_input_partial_shape(0).size() == 0)
            IE_THROW() << errorPrefix << "gets unsupported input 0D tensor (scalar)";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void ShapeOf::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    if (getParentEdges().size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

void ShapeOf::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision precision = getOriginalInputPrecisionAtPort(0);

    const LayoutType dataFormats[4] = { LayoutType::ncsp, LayoutType::nspc, LayoutType::nCsp16c, LayoutType::nCsp8c };
    for (const auto &df : dataFormats) {
        addSupportedPrimDesc({{df, precision}},
                             {{LayoutType::ncsp, Precision::I32}},
                             impl_desc_type::ref);
    }
}

bool ShapeOf::isExecutable() const {
    return true;
}

void ShapeOf::execute(dnnl::stream strm) {
    auto inPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto outPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto inDims = inPtr->getStaticDims();
    size_t dimsCount = inDims.size();
    if (outPtr->getStaticDims().size() != 1 || dimsCount != outPtr->getStaticDims()[0])
        IE_THROW() << errorPrefix << "has inconsistent input shape and output size";

    auto *dst = reinterpret_cast<int *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    for (size_t i = 0; i < dimsCount; i++) {
        dst[i] = inDims[i];
    }
}

bool ShapeOf::created() const {
    return getType() == Type::ShapeOf;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
