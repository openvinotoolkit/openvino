// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#include "kernels/x64/dft_uni_kernel.hpp"
#else
#include <assert.h>
#include <stddef.h>
struct jit_args_dft {
    const float* src;
    float* dst;
    const float* twiddles;

    size_t work_amount;
    size_t index;
};

struct jit_uni_dft_kernel {
    void (*ker_)(const jit_args_dft*);

    void operator()(const jit_args_dft* args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_dft_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_dft_kernel() {}

    virtual void create_ker() = 0;
};

struct jit_args_fft {
    const float* src;
    float* dst;
    const float* twiddles;

    size_t num_blocks;
    size_t work_amount;
    size_t n_complex;
};

struct jit_uni_fft_kernel {
    void (*ker_)(const jit_args_fft*);

    void operator()(const jit_args_fft* args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_fft_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_fft_kernel() {}

    virtual void create_ker() = 0;
};
#endif
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class DFT : public Node {
public:
    DFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    ~DFT() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    void prepareParams() override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    std::vector<int32_t> getAxes() const;
    void createJITKernels(bool hasDFT, bool hasFFT);
    void dftNd(float* output,
               const VectorDims& outputShape,
               const VectorDims& outputStrides,
               const std::vector<int32_t>& axes,
               bool inverse) const;

    void fft(float* inBuffer,
             float* outBuffer,
             int64_t dataLength,
             bool inverse,
             bool parallelize,
             const float** resultBuf) const;
    void naiveDFT(float* data, size_t dataLength, bool inverse) const;

    std::vector<float> generateTwiddlesDFT(size_t n_complex, bool inverse) const;
    void updateTwiddlesFFT(size_t n_complex, bool inverse);

    std::unique_ptr<jit_uni_dft_kernel> dftKernel = nullptr;
    std::unique_ptr<jit_uni_fft_kernel> fftKernel = nullptr;

    std::vector<float> twiddlesFFT;
    std::unordered_map<size_t, std::vector<float>> twiddlesMapDFT;

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
