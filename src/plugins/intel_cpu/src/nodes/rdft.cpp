// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rdft.h"

#include <cmath>
#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/irdft.hpp>
#include <openvino/op/rdft.hpp>
#include <string>
#include <vector>

#include "common/cpu_memcpy.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::node {

static constexpr size_t DATA_INDEX = 0;
static constexpr size_t AXES_INDEX = 1;
static constexpr size_t SIGNAL_SIZE_INDEX = 2;
static constexpr double PI = 3.14159265358979323846;

bool RDFT::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const bool isRDFT = is_type<const ov::op::v9::RDFT>(op);
        const bool isIRDFT = is_type<const ov::op::v9::IRDFT>(op);

        if (!isRDFT && !isIRDFT) {
            errorMessage = "Only opset9 RDFT/IRDFT operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

static void normalizeAxes(std::vector<int>& axes, size_t rank) {
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
    }
}

static std::vector<int> getDefaultSignalSizes(const VectorDims& inputShape,
                                              const std::vector<int>& axes,
                                              bool inverse) {
    std::vector<int> signalSizes;
    signalSizes.reserve(axes.size());

    for (auto axis : axes) {
        if (inputShape[axis] == Shape::UNDEFINED_DIM) {
            return {};
        }
        signalSizes.push_back(inputShape[axis]);
    }
    if (inverse) {
        signalSizes[signalSizes.size() - 1] = 2 * (inputShape[axes.back()] - 1);
    }

    return signalSizes;
}

RDFT::RDFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const size_t numInputs = getOriginalInputsNumber();
    if (numInputs != 2 && numInputs != 3) {
        THROW_CPU_NODE_ERR("has invalid number of input/output edges: ", numInputs);
    }

    const auto axesRank = inputShapes[AXES_INDEX].getRank();
    if (axesRank != 1) {
        THROW_CPU_NODE_ERR("has invalid 'axes' input tensor with rank: ", axesRank);
    }

    inverse = ov::is_type<ov::op::v9::IRDFT>(op);

    auto axesNode = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(1));
    if (axesNode) {
        axes = axesNode->cast_vector<int>();
        isAxesConstant = true;
        auto rank = inputShapes[DATA_INDEX].getRank() - inverse;
        normalizeAxes(axes, rank);
    }

    if (numInputs > 2) {
        const auto signalSizeRank = inputShapes[SIGNAL_SIZE_INDEX].getRank();
        if (signalSizeRank != 1) {
            THROW_CPU_NODE_ERR("has invalid 'signalSize' input tensor with rank: ", signalSizeRank);
        }
        auto signalSizesNode = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(2));
        if (!signalSizesNode) {
            return;
        }
        isSignalSizesConstant = true;
        signalSizes = signalSizesNode->cast_vector<int>();
    } else if (isAxesConstant) {
        const auto& inputShape = inputShapes[DATA_INDEX].getDims();
        signalSizes = getDefaultSignalSizes(inputShape, axes, inverse);
    }
}

void RDFT::getSupportedDescriptors() {}

void RDFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const auto& dataPrecision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    if (!dataPrecision.is_real()) {
        THROW_CPU_NODE_ERR("has unsupported 'data' input precision: ", dataPrecision.get_type_name());
    }

    const auto& axesPrecision = getOriginalInputPrecisionAtPort(AXES_INDEX);
    if (axesPrecision != ov::element::i32 && axesPrecision != ov::element::i64) {
        THROW_CPU_NODE_ERR("has unsupported 'axes' input precision: ", axesPrecision.get_type_name());
    }

    if (inputShapes.size() > SIGNAL_SIZE_INDEX) {
        const auto& signalSizePrecision = getOriginalInputPrecisionAtPort(SIGNAL_SIZE_INDEX);
        if (signalSizePrecision != ov::element::i32 && signalSizePrecision != ov::element::i64) {
            THROW_CPU_NODE_ERR("has unsupported 'signalSize' input precision: ", signalSizePrecision.get_type_name());
        }
    }

    std::vector<PortConfigurator> configurators(
        {{LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::i32}});
    if (inputShapes.size() > SIGNAL_SIZE_INDEX) {
        configurators.emplace_back(LayoutType::ncsp, ov::element::i32);
    }

    addSupportedPrimDesc(configurators, {{LayoutType::ncsp, ov::element::f32}}, impl_desc_type::ref_any);
}

void RDFT::execute(const dnnl::stream& strm) {
    const auto& inputMem = getParentEdgeAt(DATA_INDEX)->getMemory();
    const auto& outputMem = getChildEdgeAt(0)->getMemory();
    const auto& inputShape = inputMem.getStaticDims();
    const auto& outputShape = outputMem.getStaticDims();

    auto inputPtr = inputMem.getDataAs<float>();
    auto outputPtr = outputMem.getDataAs<float>();

    auto rank = inputShape.size() - inverse;

    const auto& inputStrides = inputMem.getDescWithType<BlockedMemoryDesc>()->getStrides();
    const auto& outputStrides = outputMem.getDescWithType<BlockedMemoryDesc>()->getStrides();

    executor->execute(inputPtr,
                      outputPtr,
                      twiddles,
                      rank,
                      axes,
                      signalSizes,
                      inputShape,
                      outputShape,
                      inputStrides,
                      outputStrides);
}

void RDFT::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool RDFT::created() const {
    return getType() == Type::RDFT;
}

void RDFT::prepareParams() {
    if (axesChanged()) {
        const auto& axesMem = getSrcMemoryAtPort(AXES_INDEX);
        auto newAxesSize = axesMem->getStaticDims()[0];
        if (axes.size() != newAxesSize) {
            axes.resize(newAxesSize);
        }
        auto axesPtr = axesMem->getDataAs<const int>();
        auto inputRank = inputShapes[DATA_INDEX].getRank() - inverse;
        for (size_t i = 0; i < axes.size(); i++) {
            axes[i] = axesPtr[i] < 0 ? axesPtr[i] + inputRank : axesPtr[i];
        }
    }
    if (signalSizesChanged()) {
        if (getOriginalInputsNumber() <= SIGNAL_SIZE_INDEX) {
            if (signalSizes.size() != axes.size()) {
                signalSizes.resize(axes.size());
            }
            const auto& inputShape = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims();
            for (size_t i = 0; i < axes.size() - 1; i++) {
                signalSizes[i] = inputShape[axes[i]];
            }
            if (inverse) {
                signalSizes.back() = 2 * (inputShape[axes.back()] - 1);
            } else {
                signalSizes.back() = inputShape[axes.back()];
            }
        } else {
            const auto& signalSizesMem = getSrcMemoryAtPort(SIGNAL_SIZE_INDEX);
            auto newSize = signalSizesMem->getStaticDims()[0];
            if (signalSizes.size() != newSize) {
                signalSizes.resize(newSize);
            }
            const auto& signalSizesPtr = signalSizesMem->getDataAs<const int>();
            for (size_t i = 0; i < newSize; i++) {
                signalSizes[i] = signalSizesPtr[i];
            }
        }
    }

    const auto& outputShape = getChildEdgeAt(0)->getMemory().getStaticDims();
    twiddles = executor->generateTwiddles(signalSizes, outputShape, axes);
}

bool RDFT::axesChanged() const {
    if (isAxesConstant) {
        return false;
    }
    const auto& axesMem = getSrcMemoryAtPort(AXES_INDEX);
    if (axes.size() != axesMem->getStaticDims()[0]) {
        return true;
    }
    auto axesPtr = axesMem->getDataAs<const int>();
    auto inputRank = inputShapes[DATA_INDEX].getRank() - inverse;
    for (size_t i = 0; i < axes.size(); i++) {
        auto newAxis = axesPtr[i] < 0 ? axesPtr[i] + inputRank : axesPtr[i];
        if (static_cast<size_t>(axes[i]) != newAxis) {
            return true;
        }
    }
    return false;
}

bool RDFT::signalSizesChanged() const {
    if (isSignalSizesConstant) {
        return false;
    }
    // signal sizes must have been changed if axes rank changed
    if (signalSizes.size() != axes.size()) {
        return true;
    }

    if (getOriginalInputsNumber() <= SIGNAL_SIZE_INDEX) {
        const auto& inputShape = getParentEdgeAt(DATA_INDEX)->getMemory().getStaticDims();
        for (size_t i = 0; i < axes.size() - 1; i++) {
            if (static_cast<size_t>(signalSizes[i]) != inputShape[axes[i]]) {
                return true;
            }
        }
        return inverse ? static_cast<size_t>(signalSizes.back()) != 2 * (inputShape[axes.back()] - 1)
                       : static_cast<size_t>(signalSizes.back()) != inputShape[axes.back()];
    }
    const auto& signalSizesMem = getSrcMemoryAtPort(SIGNAL_SIZE_INDEX);
    auto newSize = signalSizesMem->getStaticDims()[0];
    if (signalSizes.size() != newSize || signalSizes.size() != axes.size()) {
        return true;
    }
    const auto& signalSizesPtr = signalSizesMem->getDataAs<const int>();
    for (size_t i = 0; i < newSize; i++) {
        if (signalSizesPtr[i] != signalSizes[i]) {
            return true;
        }
    }
    return false;
}

bool RDFT::needShapeInfer() const {
    return Node::needShapeInfer() || axesChanged() || signalSizesChanged();
}

bool RDFT::needPrepareParams() const {
    return axesChanged() || signalSizesChanged() || twiddles.size() == 0;
}

static void adjustInputSize(VectorDims& inputShape,
                            std::vector<int>& signalSizes,
                            const VectorDims& outputShape,
                            const std::vector<int>& axes,
                            bool isInverse) {
    for (size_t i = 0; i < axes.size(); i++) {
        auto axis = axes[i];
        size_t inputSize = inputShape[axis];
        size_t signalSize = signalSizes[i];
        if (signalSize <= inputSize) {
            inputShape[axis] = signalSize;
        } else if (!isInverse) {
            OPENVINO_THROW("Signal size greater than input size is not supported yet");
        }
    }
    if (isInverse) {
        inputShape[axes.back()] = signalSizes.back() / 2 + 1;
    }
}

void RDFTExecutor::execute(float* inputPtr,
                           float* outputPtr,
                           const std::vector<std::vector<float>>& twiddles,
                           size_t rank,
                           const std::vector<int>& axes,
                           std::vector<int> signalSizes,
                           VectorDims inputShape,
                           const VectorDims& outputShape,
                           const VectorDims& inputStrides,
                           const VectorDims& outputStrides) {
    adjustInputSize(inputShape, signalSizes, outputShape, axes, isInverse);

    if (rank == 1) {
        auto twiddlesPtr = twiddles[0].data();
        dftCommon(inputPtr,
                  twiddlesPtr,
                  outputPtr,
                  inputShape[0],
                  signalSizes[0],
                  outputShape[0],
                  isInverse ? complex_to_real : real_to_complex,
                  canUseFFT(signalSizes[0]),
                  false);
    } else {
        if (!isInverse) {
            rdftNd(inputPtr,
                   outputPtr,
                   twiddles,
                   axes,
                   signalSizes,
                   inputShape,
                   inputStrides,
                   outputShape,
                   outputStrides);
        } else {
            irdftNd(inputPtr,
                    outputPtr,
                    twiddles,
                    axes,
                    signalSizes,
                    inputShape,
                    inputStrides,
                    outputShape,
                    outputStrides);
        }
    }
}

static void coordsFromIndex(size_t index,
                            std::vector<size_t>& coords,
                            const std::vector<size_t>& shape,
                            int excludeAxis) {
    for (size_t i = coords.size(); i > 0; i--) {
        if (static_cast<size_t>(excludeAxis) == i - 1) {
            coords[i - 1] = 0;
            continue;
        }
        coords[i - 1] = index % shape[i - 1];
        index /= shape[i - 1];
    }
}

static size_t getOffset(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    size_t offset = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

static void gatherReal(float* output,
                       const float* input,
                       size_t axis,
                       const std::vector<size_t>& coords,
                       size_t size,
                       const std::vector<size_t>& strides) {
    size_t inputOffset = getOffset(coords, strides);

    for (size_t i = 0; i < size; i++) {
        output[i] = input[inputOffset];
        inputOffset += strides[axis];
    }
}

static void gatherComplex(float* output,
                          const float* input,
                          size_t axis,
                          const std::vector<size_t>& coords,
                          size_t size,
                          const std::vector<size_t>& strides) {
    size_t inputOffset = getOffset(coords, strides);

    for (size_t i = 0; i < 2 * size; i += 2) {
        output[i] = input[inputOffset];
        output[i + 1] = input[inputOffset + 1];
        inputOffset += strides[axis];
    }
}

static void scatterReal(float* output,
                        const float* input,
                        size_t axis,
                        const std::vector<size_t>& coords,
                        size_t size,
                        const std::vector<size_t>& strides) {
    size_t offset = getOffset(coords, strides);

    for (size_t i = 0; i < size; i++) {
        output[offset] = input[i];
        offset += strides[axis];
    }
}

static void scatterComplex(float* output,
                           const float* input,
                           size_t axis,
                           const std::vector<size_t>& coords,
                           size_t size,
                           const std::vector<size_t>& strides) {
    size_t offset = getOffset(coords, strides);

    for (size_t i = 0; i < 2 * size; i += 2) {
        output[offset] = input[i];
        output[offset + 1] = input[i + 1];
        offset += strides[axis];
    }
}

static bool isPowerOfTwo(size_t n) {
    return (n != 0) && (n & (n - 1)) == 0;
}

bool RDFTExecutor::canUseFFT(size_t dim) {
    return isPowerOfTwo(dim) && dim > 1;
}

static void fftCopyInverseInputData(float* dst, float* src, size_t inputSize, size_t signalSize, bool parallelize) {
    if (!parallelize) {
        cpu_memcpy(dst, src, inputSize * complex_type_size<float>());
        src = src + 2 * inputSize - 4;
        for (size_t i = inputSize; i < signalSize; i++, src -= 2) {
            dst[2 * i] = src[0];
            dst[2 * i + 1] = -src[1];
        }
    } else {
        parallel_for(signalSize, [&](size_t i) {
            if (i < inputSize) {
                dst[2 * i] = src[2 * i];
                dst[2 * i + 1] = src[2 * i + 1];
            } else {
                size_t src_idx = 2 * inputSize - 2 - i;
                dst[2 * i] = src[2 * src_idx];
                dst[2 * i + 1] = -src[2 * src_idx + 1];
            }
        });
    }
}

static void fftCopyRealInputData(float* dst, float* src, size_t inputSize, bool parallelize) {
    if (!parallelize) {
        for (size_t i = 0; i < inputSize; i++) {
            dst[2 * i] = src[i];
            dst[2 * i + 1] = 0;
        }
    } else {
        parallel_for(inputSize, [&](size_t i) {
            dst[2 * i] = src[i];
            dst[2 * i + 1] = 0;
        });
    }
}

static void fftCopyInverseRealOutput(float* dst, float* src, size_t signalSize, bool parallelize) {
    if (!parallelize) {
        for (size_t i = 0; i < signalSize; i++) {
            dst[i] = src[2 * i];
        }
    } else {
        parallel_for(signalSize, [&](size_t i) {
            dst[i] = src[2 * i];
        });
    }
}

void RDFTExecutor::fft(float* input,
                       const float* twiddlesPtr,
                       float* output,
                       size_t inputSize,
                       size_t signalSize,
                       size_t outputSize,
                       enum dft_type type,
                       bool parallelize) {
    std::vector<float> scratchSpace(4 * signalSize, 0);

    float* inputPtr = input;
    float* outputPtr = &scratchSpace[2 * signalSize];

    if (inputSize < signalSize || type == real_to_complex) {
        if (isInverse) {
            fftCopyInverseInputData(&scratchSpace[0], input, inputSize, signalSize, parallelize);
        } else if (type == real_to_complex) {
            fftCopyRealInputData(&scratchSpace[0], input, inputSize, parallelize);
        }
        inputPtr = &scratchSpace[0];
    }

    size_t numBlocks = 0;
    size_t blockSize = 0;

    auto blockIteration = [&](size_t block) {
        size_t inputOffset = block * blockSize;
        size_t outputOffset = block * blockSize / 2;
        float cos = twiddlesPtr[2 * block];
        float sin = twiddlesPtr[2 * block + 1];
        if (isInverse) {
            sin = -sin;
        }
        for (size_t pair = 0; pair < blockSize / 2; pair++) {
            float evenReal = inputPtr[2 * (inputOffset + pair)];
            float evenImag = inputPtr[2 * (inputOffset + pair) + 1];
            float oddReal = inputPtr[2 * (inputOffset + blockSize / 2 + pair)];
            float oddImag = inputPtr[2 * (inputOffset + blockSize / 2 + pair) + 1];
            outputPtr[2 * (outputOffset + pair)] = evenReal + cos * oddReal - sin * oddImag;
            outputPtr[2 * (outputOffset + pair) + 1] = evenImag + cos * oddImag + sin * oddReal;
            outputPtr[2 * (outputOffset + signalSize / 2 + pair)] = evenReal - cos * oddReal + sin * oddImag;
            outputPtr[2 * (outputOffset + signalSize / 2 + pair) + 1] = evenImag - cos * oddImag - sin * oddReal;
            if (isInverse && numBlocks == signalSize / 2) {
                outputPtr[2 * (outputOffset + pair)] /= signalSize;
                outputPtr[2 * (outputOffset + pair) + 1] /= signalSize;
                outputPtr[2 * (outputOffset + signalSize / 2 + pair)] /= signalSize;
                outputPtr[2 * (outputOffset + signalSize / 2 + pair) + 1] /= signalSize;
            }
        }
    };

    for (numBlocks = 1; numBlocks < signalSize; numBlocks *= 2) {
        blockSize = signalSize / numBlocks;
        if (numBlocks == signalSize / 2 && outputSize == signalSize && type != complex_to_real) {
            outputPtr = output;
        }
        if (parallelize) {
            parallel_for(numBlocks, blockIteration);
        } else {
            for (size_t block = 0; block < numBlocks; block++) {
                blockIteration(block);
            }
        }
        twiddlesPtr += numBlocks * 2;
        if (numBlocks == 1 && inputPtr == input) {
            inputPtr = &scratchSpace[0];
        }
        std::swap(inputPtr, outputPtr);
    }

    if (type == complex_to_real) {
        fftCopyInverseRealOutput(output, inputPtr, signalSize, parallelize);
    } else if (outputSize != signalSize) {
        cpu_memcpy(output, inputPtr, outputSize * complex_type_size<float>());
    }
}

void RDFTExecutor::dftCommon(float* inputPtr,
                             const float* twiddlesPtr,
                             float* outputPtr,
                             size_t inputSize,
                             size_t signalSize,
                             size_t outputSize,
                             enum dft_type type,
                             bool useFFT,
                             bool parallelize) {
    if (useFFT) {
        fft(inputPtr, twiddlesPtr, outputPtr, inputSize, signalSize, outputSize, type, parallelize);
    } else {
        dft(inputPtr, twiddlesPtr, outputPtr, inputSize, signalSize, outputSize, type, parallelize);
    }
}

void RDFTExecutor::dftOnAxis(enum dft_type type,
                             float* inputPtr,
                             float* outputPtr,
                             const float* twiddlesPtr,
                             int axis,
                             size_t signalSize,
                             const VectorDims& inputShape,
                             const VectorDims& inputStrides,
                             const VectorDims& outputShape,
                             const VectorDims& outputStrides,
                             const std::vector<size_t>& iterationRange) {
    size_t inputSize = inputShape[axis];
    size_t outputSize = outputShape[axis];

    void (*gather)(float* output,
                   const float* input,
                   size_t axis,
                   const std::vector<size_t>& coords,
                   size_t size,
                   const std::vector<size_t>& strides) = nullptr;
    void (*scatter)(float* output,
                    const float* input,
                    size_t axis,
                    const std::vector<size_t>& coords,
                    size_t size,
                    const std::vector<size_t>& strides) = nullptr;

    size_t gatherSize = 0;
    size_t scatterSize = 0;

    switch (type) {
    case real_to_complex:
        scatter = scatterComplex;
        gather = gatherReal;
        gatherSize = inputSize;
        scatterSize = outputSize * 2;
        break;
    case complex_to_complex:
        gather = gatherComplex;
        scatter = scatterComplex;
        gatherSize = inputSize * 2;
        scatterSize = outputSize * 2;
        break;
    case complex_to_real:
        gather = gatherComplex;
        scatter = scatterReal;
        gatherSize = inputSize * 2;
        scatterSize = outputSize;
        break;
    }

    bool useFFT = canUseFFT(signalSize);

    size_t totalWorkSize =
        std::accumulate(iterationRange.begin(), iterationRange.end(), 1, std::multiplies<>()) / iterationRange[axis];
    bool parallelizeOuterAxes = totalWorkSize > signalSize;

    if (parallelizeOuterAxes) {
        parallel_for(totalWorkSize, [&](size_t i) {
            std::vector<size_t> coords(iterationRange.size(), 0);
            std::vector<float> gatherScatterBuffer(gatherSize + scatterSize);
            float* gatherBuffer = &gatherScatterBuffer[0];
            float* scatterBuffer = &gatherScatterBuffer[gatherSize];
            coordsFromIndex(i, coords, iterationRange, axis);
            gather(gatherBuffer, inputPtr, axis, coords, inputSize, inputStrides);
            dftCommon(gatherBuffer,
                      twiddlesPtr,
                      scatterBuffer,
                      inputSize,
                      signalSize,
                      outputSize,
                      type,
                      useFFT,
                      !parallelizeOuterAxes);
            scatter(outputPtr, scatterBuffer, axis, coords, outputSize, outputStrides);
        });
    } else {
        std::vector<size_t> coords(iterationRange.size(), 0);
        std::vector<float> gatherScatterBuffer(gatherSize + scatterSize);
        float* gatherBuffer = &gatherScatterBuffer[0];
        float* scatterBuffer = &gatherScatterBuffer[gatherSize];
        for (size_t i = 0; i < totalWorkSize; i++) {
            coordsFromIndex(i, coords, iterationRange, axis);
            gather(gatherBuffer, inputPtr, axis, coords, inputSize, inputStrides);
            dftCommon(gatherBuffer,
                      twiddlesPtr,
                      scatterBuffer,
                      inputSize,
                      signalSize,
                      outputSize,
                      type,
                      useFFT,
                      !parallelizeOuterAxes);
            scatter(outputPtr, scatterBuffer, axis, coords, outputSize, outputStrides);
        }
    }
}

// N-dimensional real DFT
void RDFTExecutor::rdftNd(float* inputPtr,
                          float* outputPtr,
                          const std::vector<std::vector<float>>& twiddles,
                          const std::vector<int>& axes,
                          const std::vector<int>& signalSizes,
                          const VectorDims& inputShape,
                          const VectorDims& inputStrides,
                          const VectorDims& outputShape,
                          const VectorDims& outputStrides) {
    const std::vector<size_t> iterationRange(outputShape.begin(), outputShape.end() - 1);

    dftOnAxis(real_to_complex,
              inputPtr,
              outputPtr,
              twiddles.back().data(),
              axes.back(),
              signalSizes.back(),
              inputShape,
              inputStrides,
              outputShape,
              outputStrides,
              iterationRange);
    inputPtr = outputPtr;

    for (size_t i = 0; i < axes.size() - 1; i++) {
        auto axis = axes[i];
        dftOnAxis(complex_to_complex,
                  inputPtr,
                  outputPtr,
                  twiddles[i].data(),
                  axis,
                  signalSizes[i],
                  outputShape,
                  outputStrides,
                  outputShape,
                  outputStrides,
                  iterationRange);
    }
}

// N-dimensional real inverse DFT
void RDFTExecutor::irdftNd(float* inputPtr,
                           float* outputPtr,
                           const std::vector<std::vector<float>>& twiddles,
                           const std::vector<int>& axes,
                           const std::vector<int>& signalSizes,
                           const VectorDims& inputShape,
                           const VectorDims& originalInputStrides,
                           const VectorDims& outputShape,
                           const VectorDims& outputStrides) {
    const std::vector<size_t> iterationRange(inputShape.begin(), inputShape.end() - 1);

    if (axes.size() == 1) {
        dftOnAxis(complex_to_real,
                  inputPtr,
                  outputPtr,
                  twiddles[0].data(),
                  axes[0],
                  signalSizes[0],
                  inputShape,
                  originalInputStrides,
                  outputShape,
                  outputStrides,
                  iterationRange);
        return;
    }

    float* output = outputPtr;
    std::vector<float> tmp;
    size_t inputShapeSize = std::accumulate(inputShape.begin(), inputShape.end(), 1, std::multiplies<>());
    size_t outputShapeSize = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<>());
    if (inputShapeSize > outputShapeSize) {
        tmp.resize(inputShapeSize);
        output = &tmp[0];
    }

    std::vector<size_t> inputStrides(originalInputStrides.size());
    inputStrides[originalInputStrides.size() - 1] = 1;
    for (size_t i = inputStrides.size() - 1; i > 0; i--) {
        inputStrides[i - 1] = inputStrides[i] * inputShape[i];
    }

    for (size_t i = 0; i < axes.size() - 1; i++) {
        auto axis = axes[i];
        dftOnAxis(complex_to_complex,
                  inputPtr,
                  output,
                  twiddles[i].data(),
                  axis,
                  signalSizes[i],
                  inputShape,
                  originalInputStrides,
                  inputShape,
                  inputStrides,
                  iterationRange);
        inputPtr = output;
    }
    dftOnAxis(complex_to_real,
              inputPtr,
              outputPtr,
              twiddles.back().data(),
              axes.back(),
              signalSizes.back(),
              inputShape,
              inputStrides,
              outputShape,
              outputStrides,
              iterationRange);
}

std::vector<float> RDFTExecutor::generateTwiddlesFFT(size_t N) {
    std::vector<float> twiddles;
    for (size_t numBlocks = 1; numBlocks < N; numBlocks *= 2) {
        for (size_t block = 0; block < numBlocks; block++) {
            double angle = 2 * PI * block / (numBlocks * 2);
            twiddles.push_back(std::cos(angle));
            twiddles.push_back(-std::sin(angle));
        }
    }
    return twiddles;
}

std::vector<float> RDFTExecutor::generateTwiddlesCommon(size_t signalSize,
                                                        size_t outputSize,
                                                        enum dft_type type,
                                                        bool useFFT) {
    if (useFFT) {
        return generateTwiddlesFFT(signalSize);
    }
    return generateTwiddlesDFT(signalSize, outputSize, type);
}

std::vector<std::vector<float>> RDFTExecutor::generateTwiddles(const std::vector<int>& signalSizes,
                                                               const std::vector<size_t>& outputShape,
                                                               const std::vector<int>& axes) {
    std::vector<std::vector<float>> twiddles;
    twiddles.reserve(axes.size());
    for (size_t i = 0; i < axes.size(); i++) {
        auto axis = axes[i];
        size_t N = signalSizes[i];
        size_t K = outputShape[axis];
        auto type = complex_to_complex;
        if (i == axes.size() - 1) {
            type = isInverse ? complex_to_real : real_to_complex;
        }
        twiddles.push_back(generateTwiddlesCommon(N, K, type, canUseFFT(N)));
    }
    return twiddles;
}
#if defined(OPENVINO_ARCH_X86_64)
struct RDFTJitExecutor : public RDFTExecutor {
    RDFTJitExecutor(bool inverse, NodeDesc* primDesc) : RDFTExecutor(inverse) {
        enum dft_type rdftType = isInverse ? complex_to_real : real_to_complex;
        if (mayiuse(cpu::x64::avx512_core)) {
            rdftKernel = std::make_unique<jit_dft_kernel_f32<cpu::x64::avx512_core>>(isInverse, rdftType);
            dftKernel = std::make_unique<jit_dft_kernel_f32<cpu::x64::avx512_core>>(isInverse, complex_to_complex);
            vlen = cpu_isa_traits<cpu::x64::avx512_core>::vlen;
            if (primDesc) {
                primDesc->setImplementationType(jit_avx512);
            }
        } else if (mayiuse(cpu::x64::avx2)) {
            rdftKernel = std::make_unique<jit_dft_kernel_f32<cpu::x64::avx2>>(isInverse, rdftType);
            dftKernel = std::make_unique<jit_dft_kernel_f32<cpu::x64::avx2>>(isInverse, complex_to_complex);
            vlen = cpu_isa_traits<cpu::x64::avx2>::vlen;
            if (primDesc) {
                primDesc->setImplementationType(jit_avx2);
            }
        } else if (mayiuse(cpu::x64::sse41)) {
            rdftKernel = std::make_unique<jit_dft_kernel_f32<cpu::x64::sse41>>(isInverse, rdftType);
            dftKernel = std::make_unique<jit_dft_kernel_f32<cpu::x64::sse41>>(isInverse, complex_to_complex);
            vlen = cpu_isa_traits<cpu::x64::sse41>::vlen;
            if (primDesc) {
                primDesc->setImplementationType(jit_sse42);
            }
        } else {
            OPENVINO_THROW("Can't create RDFT kernel");
        }

        if (rdftKernel) {
            rdftKernel->create_ker();
        }
        if (dftKernel) {
            dftKernel->create_ker();
        }
    }

    std::vector<float> generateTwiddlesDFT(size_t inputSize, size_t outputSize, enum dft_type type) override {
        std::vector<float> twiddles(inputSize * outputSize * 2);
        int simdSize = vlen / sizeof(float);
        if (type == real_to_complex) {
            simdSize /= 2;  // there are two floats per one complex element in the output
        }

        parallel_for2d(outputSize / simdSize, inputSize, [&](size_t K, size_t n) {
            if (type == real_to_complex) {
                for (int k = 0; k < simdSize; k++) {
                    double angle = 2 * PI * (K * simdSize + k) * n / inputSize;
                    twiddles[((K * inputSize + n) * simdSize + k) * 2] = std::cos(angle);
                    twiddles[((K * inputSize + n) * simdSize + k) * 2 + 1] = -std::sin(angle);
                }
            } else if (type == complex_to_real || type == complex_to_complex) {
                for (int k = 0; k < simdSize; k++) {
                    double angle = 2 * PI * (K * simdSize + k) * n / inputSize;
                    twiddles[(K * inputSize + n) * 2 * simdSize + k] = std::cos(angle);
                }
                for (int k = 0; k < simdSize; k++) {
                    double angle = 2 * PI * (K * simdSize + k) * n / inputSize;
                    twiddles[((K * inputSize + n) * 2 + 1) * simdSize + k] =
                        isInverse ? std::sin(angle) : -std::sin(angle);
                }
            }
        });
        if ((outputSize % simdSize) != 0) {
            size_t start = (outputSize / simdSize) * simdSize;
            parallel_for2d(outputSize - start, inputSize, [&](size_t k, size_t n) {
                k += start;
                double angle = 2 * PI * k * n / inputSize;
                twiddles[2 * (k * inputSize + n)] = std::cos(angle);
                twiddles[2 * (k * inputSize + n) + 1] = isInverse ? std::sin(angle) : -std::sin(angle);
            });
        }
        return twiddles;
    }

    void dft(float* inputPtr,
             const float* twiddlesPtr,
             float* outputPtr,
             size_t inputSize,
             size_t signalSize,
             size_t outputSize,
             enum dft_type type,
             bool parallelize) override {
        jit_dft_kernel* kernel = type == complex_to_complex ? dftKernel.get() : rdftKernel.get();
        if (parallelize) {
            const int cachelineSize = 64;
            size_t blockSize = 4 * cachelineSize / sizeof(float);
            size_t numBlocks = (outputSize + blockSize - 1) / blockSize;
            parallel_nt(numBlocks, [&](size_t i, size_t nthr) {
                if (numBlocks > nthr) {
                    auto newBlockSize = (((outputSize / nthr) + blockSize - 1) / blockSize) * blockSize;
                    blockSize = newBlockSize;
                    numBlocks = nthr;
                }
                jit_dft_args args{};
                args.input = inputPtr, args.twiddles = twiddlesPtr, args.output = outputPtr,
                args.input_size = inputSize, args.signal_size = signalSize, args.output_start = i * blockSize,
                args.output_end = std::min(outputSize - i * blockSize, blockSize), (*kernel)(&args);
            });
        } else {
            jit_dft_args args{};
            args.input = inputPtr, args.twiddles = twiddlesPtr, args.output = outputPtr, args.input_size = inputSize,
            args.signal_size = signalSize, args.output_start = 0, args.output_end = outputSize, (*kernel)(&args);
        }
    }

    std::unique_ptr<jit_dft_kernel> rdftKernel = nullptr;
    std::unique_ptr<jit_dft_kernel> dftKernel = nullptr;

    int vlen;
};
#endif

struct RDFTRefExecutor : public RDFTExecutor {
    RDFTRefExecutor(bool inverse) : RDFTExecutor(inverse) {}

private:
    std::vector<float> generateTwiddlesDFT(size_t inputSize, size_t outputSize, enum dft_type type) override {
        std::vector<float> twiddles(inputSize * outputSize * 2);
        parallel_for2d(outputSize, inputSize, [&](size_t k, size_t n) {
            double angle = 2 * PI * k * n / inputSize;
            if (!isInverse) {
                angle = -angle;
            }
            twiddles[(k * inputSize + n) * 2] = std::cos(angle);
            twiddles[(k * inputSize + n) * 2 + 1] = std::sin(angle);
        });
        return twiddles;
    }

    void dftRealToComplex(float* inputPtr,
                          const float* twiddlesPtr,
                          float* outputPtr,
                          size_t inputSize,
                          size_t outputSize,
                          bool parallelize) {
        auto dftIteration = [&](size_t k) {
            float real = 0, imag = 0;
            for (size_t n = 0; n < inputSize; n++) {
                float cos = twiddlesPtr[2 * (k * inputSize + n)];
                float sin = twiddlesPtr[2 * (k * inputSize + n) + 1];
                real += inputPtr[n] * cos;
                imag += inputPtr[n] * sin;
            }
            outputPtr[2 * k] = real;
            outputPtr[2 * k + 1] = imag;
        };
        if (parallelize) {
            parallel_for(outputSize, dftIteration);
        } else {
            for (size_t k = 0; k < outputSize; k++) {
                dftIteration(k);
            }
        }
    }

    void dftComplexToComplex(float* inputPtr,
                             const float* twiddlesPtr,
                             float* outputPtr,
                             size_t inputSize,
                             size_t signalSize,
                             size_t outputSize,
                             bool parallelize) {
        auto dftIteration = [&](size_t k) {
            float real = 0, imag = 0;
            for (size_t n = 0; n < inputSize; n++) {
                float cos = twiddlesPtr[2 * (k * outputSize + n)];
                float sin = twiddlesPtr[2 * (k * outputSize + n) + 1];
                float inputReal = inputPtr[2 * n];
                float inputImag = inputPtr[2 * n + 1];
                real += inputReal * cos - inputImag * sin;
                imag += inputImag * cos + inputReal * sin;
            }
            if (isInverse) {
                float* inp = inputPtr + 2 * (inputSize - 2 + outputSize % 2);
                for (size_t n = inputSize; n < signalSize; n++, inp -= 2) {
                    float cos = twiddlesPtr[2 * (k * outputSize + n)];
                    float sin = twiddlesPtr[2 * (k * outputSize + n) + 1];
                    float inputReal = inp[0];
                    float inputImag = -inp[1];
                    real += inputReal * cos - inputImag * sin;
                    imag += inputImag * cos + inputReal * sin;
                }
                real /= outputSize;
                imag /= outputSize;
            }
            outputPtr[2 * k] = real;
            outputPtr[2 * k + 1] = imag;
        };
        if (parallelize) {
            parallel_for(outputSize, dftIteration);
        } else {
            for (size_t k = 0; k < outputSize; k++) {
                dftIteration(k);
            }
        }
    }

    void dftComplexToReal(float* inputPtr,
                          const float* twiddlesPtr,
                          float* outputPtr,
                          size_t inputSize,
                          size_t signalSize,
                          size_t outputSize,
                          bool parallelize) {
        auto dftIteration = [&](size_t k) {
            float real = 0;
            for (size_t n = 0; n < inputSize; n++) {
                float cos = twiddlesPtr[2 * (k * outputSize + n)];
                float sin = twiddlesPtr[2 * (k * outputSize + n) + 1];
                float inputReal = inputPtr[2 * n];
                float inputImag = inputPtr[2 * n + 1];
                real += inputReal * cos - inputImag * sin;
            }
            if (isInverse) {
                float* inp = inputPtr + 2 * (inputSize - 2 + outputSize % 2);
                for (size_t n = inputSize; n < signalSize; n++, inp -= 2) {
                    float cos = twiddlesPtr[2 * (k * outputSize + n)];
                    float sin = twiddlesPtr[2 * (k * outputSize + n) + 1];
                    float inputReal = inp[0];
                    float inputImag = inp[1];
                    real += inputReal * cos + inputImag * sin;
                }
                real /= outputSize;
            }
            outputPtr[k] = real;
        };
        if (parallelize) {
            parallel_for(outputSize, dftIteration);
        } else {
            for (size_t k = 0; k < outputSize; k++) {
                dftIteration(k);
            }
        }
    }

    void dft(float* inputPtr,
             const float* twiddlesPtr,
             float* outputPtr,
             size_t inputSize,
             size_t signalSize,
             size_t outputSize,
             enum dft_type type,
             bool parallelize) override {
        if (type == real_to_complex) {
            dftRealToComplex(inputPtr, twiddlesPtr, outputPtr, inputSize, outputSize, parallelize);
        } else if (type == complex_to_complex) {
            dftComplexToComplex(inputPtr, twiddlesPtr, outputPtr, inputSize, signalSize, outputSize, parallelize);
        } else if (type == complex_to_real) {
            dftComplexToReal(inputPtr, twiddlesPtr, outputPtr, inputSize, signalSize, outputSize, parallelize);
        }
    }
};

void RDFT::createPrimitive() {
    RDFTKey key{};
    key.isInverse = inverse;

    auto buildExecutor = [&](const RDFTKey& key) -> std::shared_ptr<RDFTExecutor> {
        std::shared_ptr<RDFTExecutor> executor;
        NodeDesc* primDesc = getSelectedPrimitiveDescriptor();
#if defined(OPENVINO_ARCH_X86_64)
        if (mayiuse(cpu::x64::sse41)) {
            executor = std::make_shared<RDFTJitExecutor>(key.isInverse, primDesc);
            return executor;
        }
#endif
        executor = std::make_shared<RDFTRefExecutor>(key.isInverse);
        primDesc->setImplementationType(ref_any);
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    executor = result.first;

    Node::createPrimitive();
}

std::shared_ptr<RDFTExecutor> RDFTExecutor::build(bool inverse, NodeDesc* primDesc) {
    std::shared_ptr<RDFTExecutor> executor;
#if defined(OPENVINO_ARCH_X86_64)
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu::x64;
    if (mayiuse(cpu::x64::sse41)) {
        executor = std::make_shared<RDFTJitExecutor>(inverse, primDesc);
        return executor;
    }
#endif
    executor = std::make_shared<RDFTRefExecutor>(inverse);
    primDesc->setImplementationType(ref_any);
    return executor;
}

}  // namespace ov::intel_cpu::node
