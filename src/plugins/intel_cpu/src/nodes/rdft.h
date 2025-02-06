// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/primitive_hashing_utils.hpp"
#include "kernels/x64/rdft_kernel.hpp"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

struct RDFTExecutor {
public:
    RDFTExecutor(bool inverse) : isInverse(inverse) {}
    virtual ~RDFTExecutor() = default;
    void execute(float* inputPtr,
                 float* outputPtr,
                 const std::vector<std::vector<float>>& twiddles,
                 size_t rank,
                 const std::vector<int>& axes,
                 std::vector<int> signalSizes,
                 VectorDims inputShape,
                 const VectorDims& outputShape,
                 const VectorDims& inputStrides,
                 const VectorDims& outputStrides);

    std::vector<std::vector<float>> generateTwiddles(const std::vector<int>& signalSizes,
                                                     const std::vector<size_t>& outputShape,
                                                     const std::vector<int>& axes);

    static std::shared_ptr<RDFTExecutor> build(bool inverse, NodeDesc* primDesc = nullptr);

protected:
    bool isInverse;

private:
    virtual bool canUseFFT(size_t dim);
    virtual void dft(float* inputPtr,
                     const float* twiddlesPtr,
                     float* outputPtr,
                     size_t inputSize,
                     size_t signalSize,
                     size_t outputSize,
                     enum dft_type type,
                     bool parallelize) = 0;
    virtual void fft(float* input,
                     const float* twiddlesPtr,
                     float* output,
                     size_t inputSize,
                     size_t signalSize,
                     size_t outputSize,
                     enum dft_type type,
                     bool parallelize);
    void dftCommon(float* inputPtr,
                   const float* twiddlesPtr,
                   float* outputPtr,
                   size_t inputSize,
                   size_t signalSize,
                   size_t outputSize,
                   enum dft_type type,
                   bool useFFT,
                   bool parallelize);
    void dftOnAxis(enum dft_type type,
                   float* inputPtr,
                   float* outputPtr,
                   const float* twiddlesPtr,
                   int axis,
                   size_t signalSize,
                   const VectorDims& inputShape,
                   const VectorDims& inputStrides,
                   const VectorDims& outputShape,
                   const VectorDims& outputStrides,
                   const std::vector<size_t>& iteration_range);
    void rdftNd(float* inputPtr,
                float* outputPtr,
                const std::vector<std::vector<float>>& twiddles,
                const std::vector<int>& axes,
                const std::vector<int>& signalSizes,
                const VectorDims& inputShape,
                const VectorDims& inputStrides,
                const VectorDims& outputShape,
                const VectorDims& outputStrides);
    void irdftNd(float* inputPtr,
                 float* outputPtr,
                 const std::vector<std::vector<float>>& twiddles,
                 const std::vector<int>& axes,
                 const std::vector<int>& signalSizes,
                 const VectorDims& inputShape,
                 const VectorDims& inputStrides,
                 const VectorDims& outputShape,
                 const VectorDims& outputStrides);
    virtual std::vector<float> generateTwiddlesDFT(size_t inputSize, size_t outputSize, enum dft_type type) = 0;
    std::vector<float> generateTwiddlesFFT(size_t N);
    std::vector<float> generateTwiddlesCommon(size_t inputSize, size_t outputSize, enum dft_type type, bool useFFT);
};

class RDFT : public Node {
public:
    RDFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool created() const override;
    void createPrimitive() override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    bool axesChanged() const;
    bool signalSizesChanged() const;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;

    bool inverse;
    std::vector<int> axes;
    std::vector<int> signalSizes;
    std::vector<std::vector<float>> twiddles;
    std::shared_ptr<RDFTExecutor> executor;
    bool isAxesConstant = false;
    bool isSignalSizesConstant = false;
};

struct RDFTKey {
    bool isInverse;

    size_t hash() const {
        size_t seed = 0;
        seed = dnnl::impl::hash_combine(seed, isInverse);
        return seed;
    }

    bool operator==(const RDFTKey& rhs) const {
        return isInverse == rhs.isInverse;
    }
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
