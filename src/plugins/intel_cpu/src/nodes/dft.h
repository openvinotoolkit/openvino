// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>

#include "kernels/dft_uni_kernel.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class DFT : public Node {
public:
    DFT(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    ~DFT() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override;
    void prepareParams() override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    struct DFTAttrs {
        bool inverse;
        bool hasFFT;
        bool hasDFT;
    };

private:
    std::vector<int32_t> getAxes() const;

    DFTAttrs interpAttrs;

    class DFTExecutor {
    public:
        DFTExecutor(const DFTAttrs&) {}

        void exec(const float* src,
                  float* dst,
                  size_t inputRank,
                  const std::vector<int32_t>& axes,
                  const VectorDims& inputShape,
                  const VectorDims& outputShape,
                  const VectorDims& inputStrides,
                  const VectorDims& outputStrides,
                  bool inverse);

        virtual ~DFTExecutor() = default;

    private:
        void dftNd(float* output,
                   const VectorDims& outputShape,
                   const VectorDims& outputStrides,
                   const std::vector<int32_t>& axes,
                   bool inverse) const;

        virtual float* fft(float* inBuffer,
                         float* outBuffer,
                         int64_t dataLength,
                         bool inverse,
                         bool parallelize = false) const = 0;
        virtual void naiveDFT(float* data, size_t dataLength, bool inverse) const = 0;

        std::vector<float> generateTwiddlesDFT(size_t n_complex) const;
        void generateTwiddlesFFT(size_t n_complex);

    protected:
        std::vector<float> twiddlesFFT;
        std::unordered_map<size_t, std::vector<float>> twiddlesMapDFT;
    };
    std::shared_ptr<DFTExecutor> execPtr = nullptr;

    class DFTJitExecutor : public DFTExecutor {
    public:
        DFTJitExecutor(const DFTAttrs& interpAttrs);

        float* fft(float* inBuffer,
                 float* outBuffer,
                 int64_t dataLength,
                 bool inverse,
                 bool parallelize = false) const override;
        void naiveDFT(float* data, size_t dataLength, bool inverse) const override;

    private:
        std::unique_ptr<jit_uni_dft_kernel> dftKernel = nullptr;
        std::unique_ptr<jit_uni_fft_kernel> fftKernel = nullptr;
    };

    class DFTRefExecutor : public DFTExecutor {
    public:
        DFTRefExecutor(const DFTAttrs& interpAttrs) : DFTExecutor(interpAttrs) {}

        float* fft(float* inBuffer,
                 float* outBuffer,
                 int64_t dataLength,
                 bool inverse,
                 bool parallelize = false) const override;
        void naiveDFT(float* data, size_t dataLength, bool inverse) const override;
    };

    std::vector<int32_t> axes;
    std::vector<size_t> inputShape;
    std::string layerErrorPrefix;
    const size_t DATA_INDEX = 0;
    const size_t AXES_INDEX = 1;
    const size_t SIGNAL_SIZE_INDEX = 2;
    static constexpr float PI = 3.141592653589793238462643f;

    bool inverse;
    bool lastInverse;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
