// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "non_zero.h"

#include <nodes/common/cpu_memcpy.h>

#include <cpu/platform.hpp>
#include <ie_parallel.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <utils/bfloat16.hpp>

using namespace dnnl::impl::cpu::platform;
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
std::vector<size_t> NonZero::getNonZeroElementsCount(const T* src, const Shape& inShape) {
    T zero = 0;
    std::vector<size_t> counts;
    size_t inSize = inShape.getElementsCount();
    size_t inRank = inShape.getRank();

    switch (inRank) {
    case 0: {
        size_t count = src[0] != zero ? 1 : 0;
        counts.push_back(count);
        break;
    }
    case 1: {
        size_t count = 0;
        for (size_t i = 0; i < inSize; i++) {
            if (src[i] != zero) {
                count++;
            }
        }
        counts.push_back(count);
        break;
    }
    default: {
        threadsCount = std::min(parallel_get_num_threads(), static_cast<int>(inSize));
        if (threadsCount == 0)
            threadsCount = 1;

        counts.resize(threadsCount);

        parallel_nt(threadsCount, [&](size_t ithr, size_t nthr) {
            for_1d(ithr, nthr, inSize, [&](size_t i) {
                if (src[i] != zero) {
                    counts[ithr]++;
                }
            });
        });
        break;
    }
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
    const T zero = 0;
    const T *src = reinterpret_cast<T *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    Shape inShape = getParentEdgeAt(0)->getMemory().GetShape();
    size_t inRank = inShape.getRank();
    std::vector<size_t> nonZeroCounts = getNonZeroElementsCount(src, inShape);
    std::vector<size_t> destIndices(nonZeroCounts.size());
    size_t totalNonZeroCount = 0;

    for (size_t i = 0; i < nonZeroCounts.size(); ++i) {
        destIndices[i] = totalNonZeroCount;
        totalNonZeroCount += nonZeroCounts[i];
    }

    if (isDynamicNode()) {
        VectorDims newDims{inRank, totalNonZeroCount};
        redefineOutputMemory({newDims});
    }
    int* dst = reinterpret_cast<int*>(dstMemPtr->GetPtr());
    if (totalNonZeroCount == 0)
        return;

    std::vector<int> srcDims(inRank);
    std::transform(inShape.getDims().begin(), inShape.getDims().end(), srcDims.begin(), [](size_t x) {
        return static_cast<int>(x);
    });

    switch (inRank) {
    case 0:
        dst[0] = 0;
        break;
    case 1: {
        size_t& outputIndex = destIndices[0];
        for (int i = 0; i < srcDims[0]; ++i) {
            if (src[i] != zero) {
                dst[outputIndex] = i;
                outputIndex++;
            }
        }
        break;
    }
    case 2: {
        parallel_nt(threadsCount, [&](size_t ithr, size_t nthr) {
            constexpr auto blockSize = get_cache_line_size() * 2;
            constexpr auto elementsStride = blockSize / sizeof(int);
            constexpr auto elementsCount = elementsStride * 2;  // elementsStride * inRank
            int cache[elementsCount];
            int counter = 0;

            size_t& outputIndex = destIndices[ithr];

            for_2d(ithr, nthr, srcDims[0], srcDims[1], [&](size_t, size_t inputIndex, int i0, int i1) {
                if (src[inputIndex] != zero) {
                    cache[counter] = i0;
                    cache[counter + elementsStride] = i1;
                    counter++;

                    if (counter >= elementsStride) {
                        cpu_memcpy(&dst[outputIndex], cache, blockSize);
                        cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], blockSize);

                        outputIndex += elementsStride;
                        counter = 0;
                    }
                }
            });

            if (counter != 0) {
                cpu_memcpy(&dst[outputIndex], cache, counter * sizeof(int));
                cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], counter * sizeof(int));
            }
        });
        break;
    }
    case 3: {
        size_t x2totalNonZeroCount = totalNonZeroCount * 2;

        parallel_nt(threadsCount, [&](size_t ithr, size_t nthr) {
            constexpr auto blockSize = get_cache_line_size() * 2;
            constexpr auto elementsStride = blockSize / sizeof(int);
            constexpr auto elementsCount = elementsStride * 3;  // elementsStride * inRank
            int cache[elementsCount];
            int counter = 0;

            size_t& outputIndex = destIndices[ithr];
            for_3d(ithr,
                   nthr,
                   srcDims[0],
                   srcDims[1],
                   srcDims[2],
                   [&](size_t, size_t inputIndex, int i0, int i1, int i2) {
                       if (src[inputIndex] != zero) {
                            cache[counter] = i0;
                            cache[counter + elementsStride] = i1;
                            cache[counter + elementsStride * 2] = i2;
                            counter++;

                            if (counter >= elementsStride) {
                                cpu_memcpy(&dst[outputIndex], cache, blockSize);
                                cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], blockSize);
                                cpu_memcpy(&dst[outputIndex + x2totalNonZeroCount], &cache[elementsStride * 2], blockSize);

                                outputIndex += elementsStride;
                                counter = 0;
                            }
                       }
                   });

            if (counter != 0) {
                const auto remainingBlockSize = counter * sizeof(int);

                cpu_memcpy(&dst[outputIndex], cache, remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + x2totalNonZeroCount], &cache[elementsStride * 2], remainingBlockSize);
            }
        });
        break;
    }
    case 4: {
        size_t x2totalNonZeroCount = totalNonZeroCount * 2;
        size_t x3totalNonZeroCount = totalNonZeroCount * 3;

        parallel_nt(threadsCount, [&](size_t ithr, size_t nthr) {
            constexpr auto blockSize = get_cache_line_size() * 2;
            constexpr auto elementsStride = blockSize / sizeof(int);
            constexpr auto elementsCount = elementsStride * 4;  // elementsStride * inRank
            int cache[elementsCount];
            int counter = 0;

            size_t& outputIndex = destIndices[ithr];
            for_4d(
                ithr,
                nthr,
                srcDims[0],
                srcDims[1],
                srcDims[2],
                srcDims[3],
                [&](size_t, size_t inputIndex, int i0, int i1, int i2, int i3) {
                    if (src[inputIndex] != zero) {
                        cache[counter] = i0;
                        cache[counter + elementsStride] = i1;
                        cache[counter + elementsStride * 2] = i2;
                        cache[counter + elementsStride * 3] = i3;
                        counter++;

                        if (counter >= elementsStride) {
                            cpu_memcpy(&dst[outputIndex], cache, blockSize);
                            cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], blockSize);
                            cpu_memcpy(&dst[outputIndex + x2totalNonZeroCount], &cache[elementsStride * 2], blockSize);
                            cpu_memcpy(&dst[outputIndex + x3totalNonZeroCount], &cache[elementsStride * 3], blockSize);

                            outputIndex += elementsStride;
                            counter = 0;
                        }
                    }
                });

            if (counter != 0) {
                const auto remainingBlockSize = counter * sizeof(int);

                cpu_memcpy(&dst[outputIndex], cache, remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + x2totalNonZeroCount], &cache[elementsStride * 2], remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + x3totalNonZeroCount], &cache[elementsStride * 3], remainingBlockSize);
            }
        });
        break;
    }
    case 5: {
        size_t x2totalNonZeroCount = totalNonZeroCount * 2;
        size_t x3totalNonZeroCount = totalNonZeroCount * 3;
        size_t x4totalNonZeroCount = totalNonZeroCount * 4;

        parallel_nt(threadsCount, [&](size_t ithr, size_t nthr) {
            constexpr auto blockSize = get_cache_line_size() * 2;
            constexpr auto elementsStride = blockSize / sizeof(int);
            constexpr auto elementsCount = elementsStride * 5;  // elementsStride * inRank
            int cache[elementsCount];
            int counter = 0;

            size_t& outputIndex = destIndices[ithr];
            for_5d(ithr,
                   nthr,
                   srcDims[0],
                   srcDims[1],
                   srcDims[2],
                   srcDims[3],
                   srcDims[4],
                   [&](size_t, size_t inputIndex, int i0, int i1, int i2, int i3, int i4) {
                        if (src[inputIndex] != zero) {
                            cache[counter] = i0;
                            cache[counter + elementsStride] = i1;
                            cache[counter + elementsStride * 2] = i2;
                            cache[counter + elementsStride * 3] = i3;
                            cache[counter + elementsStride * 4] = i4;
                            counter++;

                            if (counter >= elementsStride) {
                                cpu_memcpy(&dst[outputIndex], cache, blockSize);
                                cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], blockSize);
                                cpu_memcpy(&dst[outputIndex + x2totalNonZeroCount], &cache[elementsStride * 2], blockSize);
                                cpu_memcpy(&dst[outputIndex + x3totalNonZeroCount], &cache[elementsStride * 3], blockSize);
                                cpu_memcpy(&dst[outputIndex + x4totalNonZeroCount], &cache[elementsStride * 4], blockSize);

                                outputIndex += elementsStride;
                                counter = 0;
                            }
                        }
                   });

            if (counter != 0) {
                const auto remainingBlockSize = counter * sizeof(int);

                cpu_memcpy(&dst[outputIndex], cache, remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + totalNonZeroCount], &cache[elementsStride], remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + x2totalNonZeroCount], &cache[elementsStride * 2], remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + x3totalNonZeroCount], &cache[elementsStride * 3], remainingBlockSize);
                cpu_memcpy(&dst[outputIndex + x4totalNonZeroCount], &cache[elementsStride * 4], remainingBlockSize);
            }
        });
        break;
    }
    default: {
        size_t inSize = inShape.getElementsCount();
        auto srcStrides = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();

        parallel_nt(threadsCount, [&](size_t ithr, size_t nthr) {
            size_t& colIndex = destIndices[ithr];
            for_1d(ithr, nthr, inSize, [&](size_t, size_t i) {
                if (src[i] != zero) {
                    size_t outIndex = 0;
                    size_t temp = i;
                    for (size_t j = 0; j < inRank; j++) {
                        outIndex = j * totalNonZeroCount + colIndex;
                        dst[outIndex] = static_cast<int>(temp / srcStrides[j]);
                        temp = temp % srcStrides[j];
                    }
                    colIndex++;
                }
            });
        });
        break;
    }
    }
}

bool NonZero::created() const {
    return getType() == Type::NonZero;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
