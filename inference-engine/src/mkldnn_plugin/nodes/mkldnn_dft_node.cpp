// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>
#include <mkldnn_extension_utils.h>

#include "mkldnn_dft_node.h"
#include "ie_parallel.hpp"
#include "ie_precision.hpp"
#include "mkldnn/ie_mkldnn.h"
#include "utils/general_utils.h"
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset7.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNDFTNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto interpDFT = std::dynamic_pointer_cast<const ngraph::opset7::DFT>(op);
        const auto interpIDFT = std::dynamic_pointer_cast<const ngraph::opset7::IDFT>(op);

        if (!interpDFT && !interpIDFT) {
            errorMessage = "Only opset7 DFT/IDFT operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDFTNode::MKLDNNDFTNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
               MKLDNNNode(op, eng, cache) {
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
    inputShape = inDims[DATA_INDEX].ToSizeVector();
    if (inputShape.size() < 2) {
        IE_THROW() << layerErrorPrefix << " has invalid 'data' input tensor with rank: " << inputShape.size();
    }

    /* Axes */
    const auto axesRank = inDims[AXES_INDEX].ndims();
    if (axesRank != 1) {
        IE_THROW() << layerErrorPrefix << " has invalid 'axes' input tensor with rank: " << axesRank;
    }

    /* Signal size */
    if (inputsNumber > SIGNAL_SIZE_INDEX) {
        const auto signalSizeRank = inDims[SIGNAL_SIZE_INDEX].ndims();
        if (signalSizeRank != 1) {
            IE_THROW() << layerErrorPrefix << " has invalid 'signal_size' input tensor with rank: " << signalSizeRank;
        }
    }

    inverse = std::dynamic_pointer_cast<ngraph::opset7::DFT>(op) == nullptr;
}

void MKLDNNDFTNode::getSupportedDescriptors() {}

void MKLDNNDFTNode::initSupportedPrimitiveDescriptors() {
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

    if (getOriginalInputsNumber() > SIGNAL_SIZE_INDEX) {
        const auto& signalSizeTensorPrec = getOriginalInputPrecisionAtPort(SIGNAL_SIZE_INDEX);
        if (signalSizeTensorPrec != Precision::I32 && signalSizeTensorPrec != Precision::I64) {
            IE_THROW() << layerErrorPrefix << " has unsupported 'signal_size' input precision: " << signalSizeTensorPrec.name();
        }
    }

    std::vector<DataConfigurator> inDataConfigurators({{TensorDescCreatorTypes::ncsp, Precision::FP32},
                                                       {TensorDescCreatorTypes::ncsp, Precision::I32}});
    if (getOriginalInputsNumber() > SIGNAL_SIZE_INDEX)
        inDataConfigurators.push_back({TensorDescCreatorTypes::ncsp,  Precision::I32});

    addSupportedPrimDesc(inDataConfigurators, {{TensorDescCreatorTypes::ncsp, Precision::FP32}}, impl_desc_type::ref_any);
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
    auto totalInput = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<size_t>());
    auto totalOutput = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<size_t>());
    std::fill_n(output, totalOutput, 0);
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
    const size_t blockSize = std::accumulate(inputShape.begin() + lastChangedDim + 1, inputShape.end(), 1ul, std::multiplies<size_t>());
    const size_t blockSizeBytes = blockSize * sizeof(float);
    std::vector<size_t> iterationCounter(iterationRange.size(), 0);
    do {
        size_t offsetInput = calculateOffsetFromStrides(iterationCounter, inputStrides);
        size_t offsetOutput = calculateOffsetFromStrides(iterationCounter, outputStrides);
        cpu_memcpy(output + offsetOutput, input + offsetInput, blockSizeBytes);
    } while (copyStep(iterationCounter, iterationRange));
}

} // namespace

void MKLDNNDFTNode::execute(mkldnn::stream strm) {
    auto axesEdge = getParentEdgeAt(AXES_INDEX);
    const auto* axesStartPtr = reinterpret_cast<const int32_t*>(axesEdge->getMemoryPtr()->GetPtr());
    axes = std::vector<int32_t>(axesStartPtr, axesStartPtr + axesEdge->getDims()[0]);
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inputShape.size() - 1;
        }
    }
    std::sort(axes.begin(), axes.end());

    outputShape = getChildEdgeAt(0)->getDims().ToSizeVector();
    for (size_t axis : axes) {
        size_t nComplex = outputShape[axis];
        // FFT uses different twiddle factors
        if (twiddlesMap.find(nComplex) == twiddlesMap.end() && !IsPowerOfTwo(nComplex)) {
            twiddlesMap[nComplex] = generateTwiddles(nComplex);
        }
    }

    auto inputDataEdge = getParentEdgeAt(DATA_INDEX);
    auto outputDataEdge = getChildEdgeAt(0);
    const auto *input = reinterpret_cast<const float*>(inputDataEdge->getMemoryPtr()->GetPtr());
    auto *output = reinterpret_cast<float*>(outputDataEdge->getMemoryPtr()->GetPtr());

    auto inputStrides = inputDataEdge->getDesc().getBlockingDesc().getStrides();
    auto outputStrides = outputDataEdge->getDesc().getBlockingDesc().getStrides();
    if (inputShape != outputShape) {
        copyDataToOutputWithSignalSize(input, inputShape, inputStrides, output, outputShape, outputStrides);
    } else {
        auto totalElements = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<size_t>());
        cpu_memcpy(output, input, totalElements * sizeof(float));
    }

    // 1d case
    if (inputDataEdge->getDesc().getDims().size() == 2) {
        size_t nComplex = outputShape[0];
        if (IsPowerOfTwo(nComplex)) {
            fft(output, nComplex * 2, true);
        } else {
            naiveDFT(output, nComplex * 2);
        }
    } else {
        dftNd(output, outputStrides);
    }
}

void MKLDNNDFTNode::dftNd(float* output, const std::vector<size_t>& outputStrides) const {
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
                    std::vector<float> gatheredData(outputLen);
                    auto parallelIterationCounter = iterationCounter;
                    parallelIterationCounter[parallelDimIndex] = dim;
                    gatherToBufferND(gatheredData.data(), output, currentAxis, parallelIterationCounter, outputShape, outputStrides);
                    fft(gatheredData.data(), outputLen);
                    applyBufferND(gatheredData.data(), output, currentAxis, parallelIterationCounter, outputShape, outputStrides);
                });
                iterationCounter[parallelDimIndex] = iterationRange[parallelDimIndex] - 1;
            } while (nextIterationStep(iterationCounter, iterationRange, currentAxis));
        } else {
            std::vector<float> gatheredData(outputLen);
            do {
                gatherToBufferND(gatheredData.data(), output, currentAxis, iterationCounter, outputShape, outputStrides);
                naiveDFT(gatheredData.data(), outputLen);
                applyBufferND(gatheredData.data(), output, currentAxis, iterationCounter, outputShape, outputStrides);
            } while (nextIterationStep(iterationCounter, iterationRange, currentAxis));
        }
    }
}

/* Cooley Tukey implementation of FFT */
void MKLDNNDFTNode::fft(float* data, int64_t dataLength, bool parallelize) const {
    static int cacheSizeL3 = utils::get_cache_size(3, false);
    static int elementsPerCacheLine = cacheSizeL3 / sizeof(float);
    std::vector<float> bufferVector(dataLength * 2, 0);
    float* buffer = bufferVector.data();
    cpu_memcpy(buffer, data, dataLength * sizeof(float));

    size_t nComplex = dataLength / 2;
    float* inBufferStart = buffer + dataLength;
    float* outBufferStart = buffer;

    auto blockIteration = [&] (const size_t block, const size_t blockSize, const size_t nextIterationBlockSize, const float anglePart) {
        float* curInpBufferPtr = inBufferStart + block * blockSize;
        float* curOutBufferPtr = outBufferStart + block * nextIterationBlockSize;

        const float angle = anglePart * block;
        const float twiddleReal = std::cos(angle);
        const float twiddleImag = -std::sin(angle);
        for (int64_t pair = 0; pair < blockSize / 2; pair += 2) {
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

    for (int64_t numBlocks = 1; numBlocks < nComplex; numBlocks *= 2) {
        const float anglePart = PI / numBlocks * (inverse ? -1 : 1);

        std::swap(inBufferStart, outBufferStart);
        const int64_t blockSize = dataLength / numBlocks;
        const int64_t nextIterationBlockSize = blockSize / 2;
        if (parallelize && blockSize >= 4 * elementsPerCacheLine) {
            parallel_for(numBlocks, [&] (const size_t block) {
                blockIteration(block, blockSize, nextIterationBlockSize, anglePart);
            });
        } else {
            for (int64_t block = 0; block < numBlocks; ++block) {
                blockIteration(block, blockSize, nextIterationBlockSize, anglePart);
            }
        }
    }

    for (int64_t k = 0; k < dataLength; k++) {
        if (inverse) {
            outBufferStart[k] /= nComplex;
        }
        data[k] = outBufferStart[k];
    }
}

void MKLDNNDFTNode::naiveDFT(float* data, size_t dataLength) const {
    std::vector<float> outputBuffer(dataLength);
    const size_t nComplex = dataLength / 2;
    const auto& twiddles = twiddlesMap.find(nComplex)->second;

    parallel_for(nComplex, [&](size_t k) {
        float sumReal = 0.0f;
        float sumImag = 0.0f;
        for (size_t n = 0; n < nComplex; ++n) {
            auto it = twiddles[k * nComplex + n];
            float complexReal = it.first;
            float complexImag = it.second;

            if (inverse) {
                complexImag *= -1; // conjugate
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

std::vector<std::pair<float, float>> MKLDNNDFTNode::generateTwiddles(size_t n_complex) const {
    std::vector<std::pair<float, float>> twiddles(n_complex * n_complex);
    parallel_for(n_complex, [&](const size_t k) {
        for (size_t n = 0; n < n_complex; ++n) {
            float phase =  2.0f * PI * static_cast<float>(n * k) / static_cast<float>(n_complex);
            auto complexReal = std::cos(phase);
            auto complexImag = -std::sin(phase);
            twiddles[k * n_complex + n] = std::make_pair(complexReal, complexImag);
        }
    });
    return twiddles;
}

bool MKLDNNDFTNode::created() const {
    return getType() == DFT;
}

void MKLDNNDFTNode::createPrimitive() {}


REG_MKLDNN_PRIM_FOR(MKLDNNDFTNode, DFT)
