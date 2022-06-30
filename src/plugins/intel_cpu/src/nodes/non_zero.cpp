// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_zero.h"
#include <ngraph/opsets/opset3.hpp>
#include <utils/bfloat16.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool NonZero::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != ngraph::op::v3::NonZero::get_type_info_static()) {
            errorMessage = "Node is not an instance of NonZero from the operation set v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

NonZero::NonZero(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng,
                                     WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "NonZero layer with name '" + getName() + "' ";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (op->get_output_element_type(0) != ngraph::element::i32) {
        IE_THROW() << errorPrefix << "doesn't support demanded output precision";
    }
}

void NonZero::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    if (getParentEdges().size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (!getChildEdges().size())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

void NonZero::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto &inPrc = getOriginalInputPrecisionAtPort(0);
    if (!one_of(inPrc, Precision::FP32, Precision::BF16, Precision::I32, Precision::U32, Precision::I8,  Precision::U8)) {
        IE_THROW() << "Can't create primitive descriptor for NonZero layer with name: " << getName() << " doesn't support "
                   << inPrc.name() << " precision on 0 port";
    }

    addSupportedPrimDesc({{LayoutType::ncsp}},
                         {{LayoutType::ncsp, Precision::I32}},
                         impl_desc_type::ref);
}

template <typename T>
size_t NonZero::getNonZeroElementsCount(const T* src, const Shape& inShape) {
    T zero = 0;
    size_t count = 0;
    size_t inSize = inShape.getElementsCount();
    if (inShape.getRank() == 0) {
        if (src[0] != zero)
            count = 1;
    } else {
        for (size_t i = 0; i < inSize; i++) {
            if (src[i] != zero)
                count++;
        }
    }
    return count;
}
namespace {
struct NonZeroContext {
    NonZero &node;
};
}
template<typename T>
struct NonZero::NonZeroExecute {
    void operator()(NonZeroContext & ctx) {
        ctx.node.executeSpecified<T>();
    }
};

void NonZero::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void NonZero::execute(dnnl::stream strm) {
    auto inputPrec = getParentEdgesAtPort(0)[0]->getMemory().getDesc().getPrecision();
    NonZeroContext ctx = {*this };
    OV_SWITCH(intel_cpu, NonZeroExecute, ctx, inputPrec,
              OV_CASE(Precision::FP32, float),
              OV_CASE(Precision::BF16, bfloat16_t),
              OV_CASE(Precision::I32, int),
              OV_CASE(Precision::U32, uint32_t),
              OV_CASE(Precision::I8, int8_t),
              OV_CASE(Precision::U8, uint8_t))
}
template <typename T>
void NonZero::executeSpecified() {
    T zero = 0;
    T *src = reinterpret_cast<T *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    Shape inShape = getParentEdgeAt(0)->getMemory().GetShape();
    size_t inRank = inShape.getRank();
    size_t nonZeroCount = getNonZeroElementsCount(src, inShape);

    if (isDynamicNode()) {
        VectorDims newDims{inRank, nonZeroCount};
        redefineOutputMemory({newDims});
    }
    int *dst = reinterpret_cast<int *>(dstMemPtr->GetPtr());
    size_t inSize = inShape.getElementsCount();
    auto srcStrides = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    if (nonZeroCount == 0)
        return;
    if (inShape.getRank() == 0) {
        dst[0] = 0;
    } else {
        size_t colIndex = 0, outIndex = 0;
        for (size_t i = 0; i < inSize; i++) {
            if (src[i] != zero) {
                size_t temp = i;
                for (size_t j = 0; j < inRank; j++) {
                    outIndex = j * nonZeroCount + colIndex;
                    dst[outIndex] = temp / srcStrides[j];
                    temp = temp % srcStrides[j];
                }
                colIndex++;
            }
        }
    }
}

bool NonZero::created() const {
    return getType() == Type::NonZero;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
