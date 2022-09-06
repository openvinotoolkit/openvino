// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft.h"

#include <string>
#include <thread>
#include <vector>
#include <cmath>
#include <dnnl_extension_utils.h>

#include <common/primitive_hashing.hpp>

#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include <onednn/dnnl.h>
#include "utils/general_utils.h"
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset7.hpp>

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

namespace {
struct DFTKey {
    ov::intel_cpu::node::DFT::DFTAttrs nodeAttrs;

    size_t hash() const;
    bool operator==(const DFTKey& rhs) const;
};

size_t DFTKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, nodeAttrs.inverse);
    seed = hash_combine(seed, nodeAttrs.hasFFT);
    seed = hash_combine(seed, nodeAttrs.hasDFT);

    return seed;
}

bool DFTKey::operator==(const DFTKey& rhs) const {
    if (nodeAttrs.inverse != rhs.nodeAttrs.inverse)
        return false;
    if (nodeAttrs.hasFFT != rhs.nodeAttrs.hasFFT)
        return false;
    if (nodeAttrs.hasDFT != rhs.nodeAttrs.hasDFT)
        return false;

    return true;
}

}  // namespace


bool DFT::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        const auto interpDFT = ov::is_type<const op::v7::DFT>(op);
        const auto interpIDFT = ov::is_type<const op::v7::IDFT>(op);

        if (!interpDFT && !interpIDFT) {
            errorMessage = "Only opset7 DFT/IDFT operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

DFT::DFT(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache) :
               Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    layerErrorPrefix = "DFT layer with name '" + op->get_name() + "'";
    const size_t inputsNumber = getOriginalInputsNumber();
    if (inputsNumber != 2 && inputsNumber != 3) {
        IE_THROW() << layerErrorPrefix << " has invalid number of input/output edges: " << inputsNumber;
    }

    /* Data */
    inputShape = inputShapes[DATA_INDEX].getStaticDims();
    if (inputShape.size() < 2) {
        IE_THROW() << layerErrorPrefix << " has invalid 'data' input tensor with rank: " << inputShape.size();
    }

    /* Axes */
    const auto axesRank = inputShapes[AXES_INDEX].getRank();
    if (axesRank != 1) {
        IE_THROW() << layerErrorPrefix << " has invalid 'axes' input tensor with rank: " << axesRank;
    }

    /* Signal size */
    if (inputsNumber > SIGNAL_SIZE_INDEX) {
        const auto signalSizeRank = inputShapes[SIGNAL_SIZE_INDEX].getRank();
        if (signalSizeRank != 1) {
            IE_THROW() << layerErrorPrefix << " has invalid 'signal_size' input tensor with rank: " << signalSizeRank;
        }
    }

    inverse = !ov::is_type<op::v7::DFT>(op);
    lastInverse = !inverse;
}

void DFT::getSupportedDescriptors() {}

void DFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& dataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    if (!dataPrecision.is_float()) {
        IE_THROW() << layerErrorPrefix << " has unsupported 'data' input precision: " << dataPrecision.name();
    }

    const auto& axesPrecision = getOriginalInputPrecisionAtPort(AXES_INDEX);
    if (axesPrecision != Precision::I32 && axesPrecision != Precision::I64) {
        IE_THROW() << layerErrorPrefix << " has unsupported 'axes' input precision: " << axesPrecision.name();
    }

    if (inputShapes.size() > SIGNAL_SIZE_INDEX) {
        const auto& signalSizeTensorPrec = getOriginalInputPrecisionAtPort(SIGNAL_SIZE_INDEX);
        if (signalSizeTensorPrec != Precision::I32 && signalSizeTensorPrec != Precision::I64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'signal_size' input precision: " << signalSizeTensorPrec.name();
        }
    }

    std::vector<PortConfigurator> inDataConfigurators({{LayoutType::ncsp, Precision::FP32},
                                                       {LayoutType::ncsp, Precision::I32}});
    if (inputShapes.size() > SIGNAL_SIZE_INDEX)
        inDataConfigurators.push_back({LayoutType::ncsp,  Precision::I32});

    addSupportedPrimDesc(inDataConfigurators, {{LayoutType::ncsp, Precision::FP32}}, impl_desc_type::ref_any);
}

namespace {
inline float getRealFromComplexProd(float lhsReal, float lhsImag, float rhsReal, float rhsImag) {
    return lhsReal * rhsReal - lhsImag * rhsImag;
}

inline float getImaginaryFromComplexProd(float lhsReal, float lhsImag, float rhsReal, float rhsImag) {
    return lhsReal * rhsImag + lhsImag * rhsReal;
}

/*
    Returns true while we can iterate
    Specified axis is skipped in counters   
*/
inline bool nextIterationStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange, size_t axis) {
    auto itCounter = counters.rbegin();
    auto itWork = iterationRange.rbegin();

    while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
        if (std::distance(itCounter, counters.rend()) == axis + 1) {
            ++itCounter;
            ++itWork;
            continue;
        }
        *itCounter = (*itCounter + 1) % *itWork;
        if (*itCounter != 0) {
            return true;
        }
        ++itCounter;
        ++itWork;
    }
    return false;
}

inline bool IsPowerOfTwo(size_t n) {
    return (n != 0) && (n & (n - 1)) == 0;
}

inline bool copyStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange) {
    auto itCounter = counters.rbegin();
    auto itWork = iterationRange.rbegin();

    while (itCounter != counters.rend() && itWork != iterationRange.rend()) {
        *itCounter = (*itCounter + 1) % *itWork;
        if (*itCounter != 0) {
            return true;
        }
        ++itCounter;
        ++itWork;
    }
    return false;
}

size_t calculateOffsetFromStrides(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    size_t offset = 0;
    for (size_t index = 0; index < coords.size(); ++index) {
        offset += coords[index] * strides[index];
    }
    return offset;
}

void gatherToBufferND(float* buffer, const float* data, size_t axis, const std::vector<size_t>& dimIndexes,
                                     const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
    size_t numberOfComplex = shape[axis];
    size_t offset = calculateOffsetFromStrides(dimIndexes, strides);

    for (size_t bufferIndex = 0; bufferIndex < 2 * numberOfComplex; bufferIndex += 2) {
        buffer[bufferIndex] = data[offset];
        buffer[bufferIndex + 1] = data[offset + 1];
        offset += strides[axis];
    }
}

void applyBufferND(const float* buffer, float* output, size_t axis, const std::vector<size_t>& dimIndexes,
                                  const std::vector<size_t>& shape, const std::vector<size_t>& strides) {
    size_t numberOfComplex = shape[axis];
    size_t offset = calculateOffsetFromStrides(dimIndexes, strides);

    for (size_t bufferIndex = 0; bufferIndex < 2 * numberOfComplex; bufferIndex += 2) {
        output[offset] = buffer[bufferIndex];
        output[offset + 1] = buffer[bufferIndex + 1];
        offset += strides[axis];
    }
}

void copyDataToOutputWithSignalSize(const float* input, const std::vector<size_t>& inputShape, const std::vector<size_t>& inputStrides,
                                    float* output, const std::vector<size_t>& outputShape, const std::vector<size_t>& outputStrides) {
    auto totalInput = std::accumulate(inputShape.begin(), inputShape.end(), size_t(1), std::multiplies<size_t>());
    auto totalOutput = std::accumulate(outputShape.begin(), outputShape.end(), size_t(1), std::multiplies<size_t>());
    std::fill_n(output, totalOutput, 0.f);
    size_t lastChangedDim = 0;
    for (size_t index = inputShape.size() - 1; index > 0; --index) {
        if (inputShape[index] != outputShape[index]) {
            lastChangedDim = index;
            break;
        }
    }
    if (lastChangedDim == 0) {
        size_t outputBytesSize = std::min(totalOutput, totalInput) * sizeof(float);
        cpu_memcpy(output, input, outputBytesSize);
        return;
    }

    std::vector<size_t> iterationRange(lastChangedDim + 1, 0);
    for (size_t index = 0; index < lastChangedDim + 1; ++index) {
        iterationRange[index] = std::min(inputShape[index], outputShape[index]);
    }

    const std::vector<size_t> inputStridesRange(inputStrides.begin(), inputStrides.begin() + iterationRange.size());
    const std::vector<size_t> outputStridesRange(outputStrides.begin(), outputStrides.begin() + iterationRange.size());
    const size_t blockSize = std::accumulate(inputShape.begin() + lastChangedDim + 1, inputShape.end(), size_t(1), std::multiplies<size_t>());
    const size_t blockSizeBytes = blockSize * sizeof(float);
    std::vector<size_t> iterationCounter(iterationRange.size(), 0);
    do {
        size_t offsetInput = calculateOffsetFromStrides(iterationCounter, inputStrides);
        size_t offsetOutput = calculateOffsetFromStrides(iterationCounter, outputStrides);
        cpu_memcpy(output + offsetOutput, input + offsetInput, blockSizeBytes);
    } while (copyStep(iterationCounter, iterationRange));
}

} // namespace

void DFT::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute DFT node with name: " << getName() << ". Primitive isn't created";
        return;
    }

    const auto& outputShape = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();

    const auto inputDataEdge = getParentEdgeAt(DATA_INDEX);
    const auto outputDataEdge = getChildEdgeAt(0);

    const auto src = reinterpret_cast<const float*>(inputDataEdge->getMemoryPtr()->GetPtr());
    auto dst = reinterpret_cast<float*>(outputDataEdge->getMemoryPtr()->GetPtr());

    const auto inputRank = inputDataEdge->getMemory().GetShape().getRank();

    const auto& inputStrides = inputDataEdge->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();
    const auto& outputStrides = outputDataEdge->getMemory().GetDescWithType<BlockedMemoryDesc>()->getStrides();

    execPtr->exec(src, dst, inputRank, axes, inputShape, outputShape, inputStrides, outputStrides, inverse);
}

void DFT::DFTExecutor::exec(const float* src,
                            float* dst,
                            size_t inputRank,
                            const std::vector<int32_t>& axes,
                            const VectorDims& inputShape,
                            const VectorDims& outputShape,
                            const VectorDims& inputStrides,
                            const VectorDims& outputStrides,
                            bool inverse) {
    size_t nComplexMaxFFT = 0;
    for (size_t axis : axes) {
        size_t nComplex = outputShape[axis];
        // FFT uses different twiddle factors
        if (!IsPowerOfTwo(nComplex)) {
            if (twiddlesMapDFT.find(nComplex) == twiddlesMapDFT.end()) {
                twiddlesMapDFT[nComplex] = generateTwiddlesDFT(nComplex);
            }
        } else {
            if (nComplexMaxFFT < nComplex) {
                nComplexMaxFFT = nComplex;
            }
        }
    }

    if (nComplexMaxFFT > 0 && (nComplexMaxFFT - 1) * 2 > twiddlesFFT.size()) {
        generateTwiddlesFFT(nComplexMaxFFT);
    }

    if (inputShape != outputShape) {
        copyDataToOutputWithSignalSize(src, inputShape, inputStrides, dst, outputShape, outputStrides);
    } else {
        auto totalElements = std::accumulate(inputShape.begin(), inputShape.end(), size_t(1), std::multiplies<size_t>());
        cpu_memcpy(dst, src, totalElements * sizeof(float));
    }

    // 1d case
    if (inputRank == 2) {
        size_t nComplex = outputShape[0];
        if (IsPowerOfTwo(nComplex)) {
            std::vector<float> outputData(nComplex * 2);
            const auto* outPtr = fft(dst, outputData.data(), nComplex * 2, inverse, true);

            if (outPtr != dst) {
                cpu_memcpy(dst, outPtr, nComplex * 2 * sizeof(float));
            }
        } else {
            naiveDFT(dst, nComplex * 2, inverse);
        }
    } else {
        dftNd(dst, outputShape, outputStrides, axes, inverse);
    }
}

void DFT::DFTExecutor::dftNd(float* output,
                             const VectorDims& outputShape,
                             const VectorDims& outputStrides,
                             const std::vector<int32_t>& axes,
                             bool inverse) const {
    const std::vector<size_t> iterationRange(outputShape.begin(), outputShape.end() - 1);
    const size_t lastDimIndex = iterationRange.size() - 1;
    for (size_t axisIndex = 0; axisIndex < axes.size(); ++axisIndex) {
        const size_t currentAxis = axes[axisIndex];
        const size_t outputComplexLen = outputShape[currentAxis];
        const size_t outputLen = outputComplexLen * 2;

        std::vector<size_t> iterationCounter(iterationRange.size(), 0);
        if (IsPowerOfTwo(outputComplexLen)) {
            size_t parallelDimIndex = lastDimIndex == currentAxis ? lastDimIndex - 1 : lastDimIndex;
            do {
                parallel_for(iterationRange[parallelDimIndex], [&](size_t dim) {
                    std::vector<float> gatheredData(outputLen * 2);
                    auto parallelIterationCounter = iterationCounter;
                    parallelIterationCounter[parallelDimIndex] = dim;
                    gatherToBufferND(gatheredData.data(), output, currentAxis, parallelIterationCounter, outputShape, outputStrides);
                    const auto* bufferPtr =
                        fft(gatheredData.data(), gatheredData.data() + outputLen, outputLen, inverse);
                    applyBufferND(bufferPtr, output, currentAxis, parallelIterationCounter, outputShape, outputStrides);
                });
                iterationCounter[parallelDimIndex] = iterationRange[parallelDimIndex] - 1;
            } while (nextIterationStep(iterationCounter, iterationRange, currentAxis));
        } else {
            std::vector<float> gatheredData(outputLen);
            do {
                gatherToBufferND(gatheredData.data(), output, currentAxis, iterationCounter, outputShape, outputStrides);
                naiveDFT(gatheredData.data(), outputLen, inverse);
                applyBufferND(gatheredData.data(), output, currentAxis, iterationCounter, outputShape, outputStrides);
            } while (nextIterationStep(iterationCounter, iterationRange, currentAxis));
        }
    }
}

/* Cooley Tukey implementation of FFT */
float* DFT::DFTRefExecutor::fft(float* inBuffer,
                                float* outBuffer,
                                int64_t dataLength,
                                bool inverse,
                                bool parallelize) const {
    static int cacheSizeL3 = dnnl::utils::get_cache_size(3, false);
    static int elementsPerCacheLine = cacheSizeL3 / sizeof(float);
    size_t nComplex = dataLength / 2;

    auto blockIteration =
        [&](const size_t numBlocks, const size_t block, const size_t blockSize, const size_t nextIterationBlockSize) {
            float* curInpBufferPtr = inBuffer + block * blockSize;
            float* curOutBufferPtr = outBuffer + block * nextIterationBlockSize;
            float twiddleReal = twiddlesFFT[(numBlocks + block - 1) * 2];
            float twiddleImag = twiddlesFFT[(numBlocks + block) * 2 - 1];

            if (inverse) {
                twiddleImag *= -1;
            }

            for (size_t pair = 0; pair < blockSize / 2; pair += 2) {
                const float evenReal = curInpBufferPtr[pair];
                const float evenImag = curInpBufferPtr[pair + 1];

                const float oddReal = curInpBufferPtr[(blockSize / 2 + pair)];
                const float oddImag = curInpBufferPtr[(blockSize / 2 + pair) + 1];

                const float twiddledOddReal = getRealFromComplexProd(twiddleReal, twiddleImag, oddReal, oddImag);
                const float twiddledOddImag = getImaginaryFromComplexProd(twiddleReal, twiddleImag, oddReal, oddImag);

                curOutBufferPtr[pair] = evenReal + twiddledOddReal;
                curOutBufferPtr[pair + 1] = evenImag + twiddledOddImag;

                curOutBufferPtr[nComplex + pair] = evenReal - twiddledOddReal;
                curOutBufferPtr[nComplex + pair + 1] = evenImag - twiddledOddImag;
            }
        };

    size_t blockSize;
    size_t nextIterationBlockSize = dataLength;
    for (size_t numBlocks = 1; numBlocks < nComplex; numBlocks *= 2) {
        blockSize = nextIterationBlockSize;
        nextIterationBlockSize /= 2;
        if (parallelize && blockSize >= 4 * elementsPerCacheLine) {
            parallel_for(numBlocks, [&](const size_t block) {
                blockIteration(numBlocks, block, blockSize, nextIterationBlockSize);
            });
        } else {
            for (size_t block = 0; block < numBlocks; ++block) {
                blockIteration(numBlocks, block, blockSize, nextIterationBlockSize);
            }
        }

        std::swap(inBuffer, outBuffer);
    }
    if (inverse) {
        for (int64_t k = 0; k < dataLength; k++) {
            inBuffer[k] /= nComplex;
        }
    }

    return inBuffer;
}

void DFT::DFTRefExecutor::naiveDFT(float* data, size_t dataLength, bool inverse) const {
    std::vector<float> outputBuffer(dataLength);
    const size_t nComplex = dataLength / 2;
    const auto& twiddles = twiddlesMapDFT.find(nComplex)->second;

    parallel_for(nComplex, [&](size_t k) {
        float sumReal = 0.0f;
        float sumImag = 0.0f;
        for (size_t n = 0; n < nComplex; ++n) {
            auto complexRef = &twiddles[2 * (k * nComplex + n)];
            float complexReal = *complexRef;
            float complexImag = *(complexRef + 1);

            if (inverse) {
                complexImag *= -1;  // conjugate
            }
            float complexProdReal = getRealFromComplexProd(data[2 * n], data[2 * n + 1], complexReal, complexImag);
            float complexProdImag = getImaginaryFromComplexProd(data[2 * n], data[2 * n + 1], complexReal, complexImag);

            sumReal += complexProdReal;
            sumImag += complexProdImag;
        }

        if (inverse) {
            sumReal /= nComplex;
            sumImag /= nComplex;
        }
        outputBuffer[k * 2] = sumReal;
        outputBuffer[k * 2 + 1] = sumImag;
    });
    cpu_memcpy(data, outputBuffer.data(), dataLength * sizeof(float));
}

std::vector<float> DFT::DFTExecutor::generateTwiddlesDFT(size_t n_complex) const {
    std::vector<float> twiddles(n_complex * n_complex * 2);
    parallel_for(n_complex, [&](const size_t k) {
        for (size_t n = 0; n < n_complex; ++n) {
            float phase = 2.0f * PI * static_cast<float>(n * k) / static_cast<float>(n_complex);
            auto complexReal = std::cos(phase);
            auto complexImag = -std::sin(phase);
            twiddles[2 * (k * n_complex + n)] = complexReal;
            twiddles[2 * (k * n_complex + n) + 1] = complexImag;
        }
    });
    return twiddles;
}

void DFT::DFTExecutor::generateTwiddlesFFT(size_t n_complex) {
    size_t numBlocks = 1;

    twiddlesFFT.reserve((n_complex - 1) * 2);
    if (twiddlesFFT.size() == 0) {
        twiddlesFFT.emplace_back(1.0f);   //  cos(0)
        twiddlesFFT.emplace_back(-0.0f);  // -sin(0)
    } else {
        for (size_t i = numBlocks; i < twiddlesFFT.size() / 2; i += numBlocks) {
            numBlocks *= 2;
        }
    }

    for (size_t i = twiddlesFFT.size() / 2; i < n_complex - 1; i += numBlocks) {
        numBlocks *= 2;

        for (size_t blockNum = 0; blockNum < numBlocks; blockNum++) {
            size_t copyIndex = twiddlesFFT.size() - blockNum - numBlocks;

            twiddlesFFT.push_back(twiddlesFFT[copyIndex]);
            twiddlesFFT.push_back(twiddlesFFT[copyIndex + 1]);

            blockNum++;

            float angle = PI * blockNum / numBlocks;
            auto complexReal = std::cos(angle);
            auto complexImag = -std::sin(angle);

            twiddlesFFT.emplace_back(complexReal);
            twiddlesFFT.emplace_back(complexImag);
        }
    }
}

DFT::DFTJitExecutor::DFTJitExecutor(const DFTAttrs& interpAttrs) : DFTExecutor(interpAttrs) {
    jit_dft_config_params jdp = {};

    jdp.inverse = interpAttrs.inverse;

    if (interpAttrs.hasDFT) {
        if (mayiuse(cpu::x64::avx512_core)) {
            dftKernel.reset(new jit_uni_dft_kernel_f32<cpu::x64::avx512_core>(jdp));
        } else if (mayiuse(cpu::x64::avx2)) {
            dftKernel.reset(new jit_uni_dft_kernel_f32<cpu::x64::avx2>(jdp));
        } else if (mayiuse(cpu::x64::sse41)) {
            dftKernel.reset(new jit_uni_dft_kernel_f32<cpu::x64::sse41>(jdp));
        } else {
            IE_THROW() << "Can't create jit DFT kernel";
        }

        if (dftKernel)
            dftKernel->create_ker();
    }

    if (interpAttrs.hasFFT) {
        if (mayiuse(cpu::x64::avx512_core)) {
            fftKernel.reset(new jit_uni_fft_kernel_f32<cpu::x64::avx512_core>(jdp));
        } else if (mayiuse(cpu::x64::avx2)) {
            fftKernel.reset(new jit_uni_fft_kernel_f32<cpu::x64::avx2>(jdp));
        } else if (mayiuse(cpu::x64::sse41)) {
            fftKernel.reset(new jit_uni_fft_kernel_f32<cpu::x64::sse41>(jdp));
        } else {
            IE_THROW() << "Can't create jit FFT kernel";
        }

        if (fftKernel)
            fftKernel->create_ker();
    }
}

float* DFT::DFTJitExecutor::fft(float* inBuffer,
                                float* outBuffer,
                                int64_t dataLength,
                                bool inverse,
                                bool parallelize) const {
    static int cacheSizeL3 = dnnl::utils::get_cache_size(3, false);
    static int elementsPerCacheLine = cacheSizeL3 / sizeof(float);
    size_t nComplex = dataLength / 2;

    auto blockIteration =
        [&](const size_t block, const size_t numBlocks, const size_t blockSize, const size_t nextIterationBlockSize) {
            auto arg = jit_args_fft();

            arg.src = inBuffer + block * nextIterationBlockSize * 2;
            arg.dst = outBuffer + block * nextIterationBlockSize;
            arg.twiddles = &twiddlesFFT[(numBlocks + block - 1) * 2];
            arg.num_blocks = numBlocks;
            arg.work_amount = nextIterationBlockSize;
            arg.n_complex = nComplex;

            (*fftKernel)(&arg);
        };

    size_t blockSize;
    size_t nextIterationBlockSize = dataLength;
    for (size_t numBlocks = 1; numBlocks < nComplex; numBlocks *= 2) {
        blockSize = nextIterationBlockSize;
        nextIterationBlockSize /= 2;
        if (parallelize && blockSize >= 4 * elementsPerCacheLine) {
            parallel_for(numBlocks, [&](const size_t block) {
                blockIteration(block, 1, blockSize, nextIterationBlockSize);
            });
        } else {
            blockIteration(0, numBlocks, blockSize, nextIterationBlockSize);
        }

        std::swap(inBuffer, outBuffer);
    }

    if (inverse) {
        for (int64_t k = 0; k < dataLength; k++) {
            inBuffer[k] /= nComplex;
        }
    }

    return inBuffer;
}

void DFT::DFTJitExecutor::naiveDFT(float* data, size_t dataLength, bool inverse) const {
    std::vector<float> outputBuffer(dataLength);
    const size_t nComplex = dataLength / 2;
    const auto& twiddles = twiddlesMapDFT.find(nComplex)->second;

    parallel_for(nComplex, [&](size_t k) {
        auto arg = jit_args_dft();

        arg.src = data;
        arg.dst = outputBuffer.data() + 2 * k;
        arg.twiddles = twiddles.data() + 2 * k * nComplex;
        arg.work_amount = nComplex;
        arg.index = k;

        (*dftKernel)(&arg);
    });
    cpu_memcpy(data, outputBuffer.data(), dataLength * sizeof(float));
}

bool DFT::created() const {
    return getType() == Type::DFT;
}

bool DFT::needPrepareParams() const {
    return lastInverse != inverse;
}

void DFT::prepareParams() {
    DFTKey key = {};

    key.nodeAttrs.inverse = inverse;
    key.nodeAttrs.hasDFT = false;
    key.nodeAttrs.hasFFT = false;

    axes = getAxes();
    const auto outputShape = getChildEdgesAtPort(0)[0]->getMemory().getStaticDims();

    for (size_t axis : axes) {
        size_t nComplex = outputShape[axis];
        if (!IsPowerOfTwo(nComplex)) {
            key.nodeAttrs.hasDFT = true;
        } else {
            key.nodeAttrs.hasFFT = true;
        }
    }

    auto buildExecutor = [&](const DFTKey& key) -> std::shared_ptr<DFTExecutor> {
        std::shared_ptr<DFTExecutor> executor;
        if (mayiuse(cpu::x64::sse41)) {
            executor = std::make_shared<DFTJitExecutor>(key.nodeAttrs);
        } else {
            executor = std::make_shared<DFTRefExecutor>(key.nodeAttrs);
        }
        return executor;
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    execPtr = result.first;

    lastInverse = inverse;
}

std::vector<int32_t> DFT::getAxes() const {
    auto axesEdge = getParentEdgeAt(AXES_INDEX);
    const auto* axesStartPtr = reinterpret_cast<const int32_t*>(axesEdge->getMemoryPtr()->GetPtr());
    auto axes = std::vector<int32_t>(axesStartPtr, axesStartPtr + axesEdge->getMemory().getStaticDims()[0]);
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inputShape.size() - 1;
        }
    }
    std::sort(axes.begin(), axes.end());
    return axes;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
