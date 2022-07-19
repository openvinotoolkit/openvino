// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <map>
#include "kernels/rdft_kernel.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

struct RDFTExecutor {
    public:
        RDFTExecutor(bool inverse) : isInverse(inverse) {}
        void execute(float* src, float* dst, size_t rank,
                     const std::vector<int>& axes,
                     std::vector<int> signalSizes,
                     VectorDims inputShape, const VectorDims& outputShape,
                     const VectorDims& inputStrides, const VectorDims& outputStrides);

    protected:
        std::vector<std::vector<float>> twiddles;
        bool isInverse;

    private:
        virtual bool canUseFft(size_t dim) = 0;
        virtual void dft(float* inputPtr, float* twiddlesPtr, float* outputPtr,
                         size_t inputSize, size_t signalSize, size_t outputSize,
                         enum dft_type type, bool parallelize) = 0;
        virtual void fft(float* input, float* twiddlesPtr, float* output,
                         size_t inputSize, size_t signalSize, size_t outputSize,
                         enum dft_type type, bool parallelize) = 0;
        void dftCommon(float* inputPtr, float* twiddlesPtr, float* outputPtr,
                        size_t inputSize, size_t signalSize, size_t outputSize,
                        enum dft_type type, bool useFft, bool parallelize);
        void dftOnAxis(enum dft_type type,
                         float* inputPtr, float* outputPtr,
                         float* twiddlesPtr, int axis,
                         size_t signalSize,
                         const VectorDims& inputShape,
                         const VectorDims& inputStrides,
                         const VectorDims& outputShape,
                         const VectorDims& outputStrides,
                         const std::vector<size_t>& iteration_range);
        void rdftNd(float* inputPtr, float* outputPtr,
                     const std::vector<int>& axes,
                     const std::vector<int>& signalSizes,
                     const VectorDims& inputShape,
                     const VectorDims& inputStrides,
                     const VectorDims& outputShape,
                     const VectorDims& outputStrides);
        void irdftNd(float* inputPtr, float* outputPtr,
                      const std::vector<int>& axes,
                      const std::vector<int>& signalSizes,
                      const VectorDims& inputShape,
                      const VectorDims& inputStrides,
                      const VectorDims& outputShape,
                      const VectorDims& outputStrides);
        virtual std::vector<float> generateTwiddlesDft(size_t inputSize, size_t outputSize, enum dft_type type) = 0;
        std::vector<float> generateTwiddlesFft(size_t N);
        std::vector<float> generateTwiddlesCommon(size_t inputSize, size_t outputSize,
                                                  enum dft_type type, bool useFft);
        void generateTwiddles(const std::vector<int>& signalSizes,
                               const std::vector<size_t>& outputShape,
                               const std::vector<int>& axes);
};

class RDFT : public Node {
public:
    RDFT(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool needPrepareParams() const override;
    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string errorMsgPrefix;

    bool inverse;

    std::shared_ptr<RDFTExecutor> executor;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
