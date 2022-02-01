// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_zero.h"
#include <ie_parallel.hpp>
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
    if (getChildEdges().size() != 1)
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
std::vector<size_t> MKLDNNNonZeroNode::getNonZeroElementsCount(const T* src, const Shape& inShape) {
    T zero = 0;
    std::vector<size_t> counts;
    size_t inSize = inShape.getElementsCount();
    if (inShape.getRank() == 0) {
        size_t count = src[0] != zero ? 1 : 0;
        counts.push_back(count);
    } else {
        int nthr = std::min(parallel_get_num_threads(), static_cast<int>(inSize));
        if (nthr == 0)
            nthr = 1;

        counts.resize(nthr);

        parallel_for(inSize, [&](size_t ithr, size_t i) {
            if (src[i] != zero)
                counts[ithr]++;
        });
    }
    return counts;
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
    std::vector<size_t> nonZeroCounts = getNonZeroElementsCount(src, inShape);
    std::vector<size_t> destIndices(nonZeroCounts);
    size_t totalNonZeroCount = 0;

    for (size_t i = 0; i < nonZeroCounts.size(); ++i) {
        destIndices[i] = totalNonZeroCount;
        totalNonZeroCount += nonZeroCounts[i];
    }

    if (isDynamicNode()) {
        VectorDims newDims{inRank, totalNonZeroCount};
        redefineOutputMemory({newDims});
    }
    int *dst = reinterpret_cast<int *>(dstMemPtr->GetPtr());
    auto srcDims = inShape.getDims();
    if (totalNonZeroCount == 0)
        return;

    switch (inRank) {
    case 0:
        dst[0] = 0;
        break;
    case 1:
        parallel_for(srcDims[0],
                     [&](size_t ithr, size_t i0) {
                         size_t inputIndex = i0;
                         size_t& outputIndex = destIndices[ithr];

                         if (src[inputIndex] != zero) {
                             dst[outputIndex] = static_cast<int>(i0);
                             outputIndex++;
                         }
                     });
        break;
    case 2:
    {
        size_t i0Stride = srcDims[1];

        parallel_for2d(srcDims[0], srcDims[1],
                       [&](size_t ithr, size_t i0, size_t i1) {
                           size_t inputIndex = i0 * i0Stride + i1;
                           size_t& outputIndex = destIndices[ithr];

                           if (src[inputIndex] != zero) {
                               dst[outputIndex] = static_cast<int>(i0);
                               dst[outputIndex + totalNonZeroCount] = static_cast<int>(i1);
                               outputIndex++;
                           }
                       });
        break;
    }
    case 3:
    {
        size_t i1Stride = srcDims[2];
        size_t i0Stride = srcDims[1] * i1Stride;
        size_t x2totalNonZeroCount = totalNonZeroCount * 2;

        parallel_for3d(srcDims[0], srcDims[1], srcDims[2],
                       [&](size_t ithr, size_t i0, size_t i1, size_t i2) {
                           size_t inputIndex = i0 * i0Stride + i1 * i1Stride + i2;
                           size_t& outputIndex = destIndices[ithr];

                           if (src[inputIndex] != zero) {
                               dst[outputIndex] = static_cast<int>(i0);
                               dst[outputIndex + totalNonZeroCount] = static_cast<int>(i1);
                               dst[outputIndex + x2totalNonZeroCount] = static_cast<int>(i2);
                               outputIndex++;
                           }
                       });
        break;
    }
    case 4:
    {
        size_t i2Stride = srcDims[3];
        size_t i1Stride = srcDims[2] * i2Stride;
        size_t i0Stride = srcDims[1] * i1Stride;
        size_t x2totalNonZeroCount = totalNonZeroCount * 2;
        size_t x3totalNonZeroCount = totalNonZeroCount * 3;

        parallel_for4d(srcDims[0], srcDims[1], srcDims[2], srcDims[3],
                       [&](size_t ithr, size_t i0, size_t i1, size_t i2, size_t i3) {
                           size_t inputIndex = i0 * i0Stride + i1 * i1Stride + i2 * i2Stride + i3;
                           size_t& outputIndex = destIndices[ithr];

                           if (src[inputIndex] != zero) {
                               dst[outputIndex] = static_cast<int>(i0);
                               dst[outputIndex + totalNonZeroCount] = static_cast<int>(i1);
                               dst[outputIndex + x2totalNonZeroCount] = static_cast<int>(i2);
                               dst[outputIndex + x3totalNonZeroCount] = static_cast<int>(i3);
                               outputIndex++;
                           }
                       });
        break;
    }
    case 5:
    {
        size_t i3Stride = srcDims[4];
        size_t i2Stride = srcDims[3] * i3Stride;
        size_t i1Stride = srcDims[2] * i2Stride;
        size_t i0Stride = srcDims[1] * i1Stride;
        size_t x2totalNonZeroCount = totalNonZeroCount * 2;
        size_t x3totalNonZeroCount = totalNonZeroCount * 3;
        size_t x4totalNonZeroCount = totalNonZeroCount * 4;

        parallel_for5d(srcDims[0], srcDims[1], srcDims[2], srcDims[3], srcDims[4],
                       [&](size_t ithr, size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
                           size_t inputIndex = i0 * i0Stride + i1 * i1Stride + i2 * i2Stride + i3 * i3Stride + i4;
                           size_t& outputIndex = destIndices[ithr];

                           if (src[inputIndex] != zero) {
                               dst[outputIndex] = static_cast<int>(i0);
                               dst[outputIndex + totalNonZeroCount] = static_cast<int>(i1);
                               dst[outputIndex + x2totalNonZeroCount] = static_cast<int>(i2);
                               dst[outputIndex + x3totalNonZeroCount] = static_cast<int>(i3);
                               dst[outputIndex + x4totalNonZeroCount] = static_cast<int>(i4);
                               outputIndex++;
                           }
                       });
        break;
    }
    default:
        assert(false);
        break;
    }
}

bool NonZero::created() const {
    return getType() == Type::NonZero;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
