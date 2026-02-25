// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "kernels/x64/dft_uni_kernel.hpp"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class DFT : public ov::intel_cpu::Node {
public:
    DFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    ~DFT() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    void createPrimitive() override;
    [[nodiscard]] bool needShapeInfer() const override;
    [[nodiscard]] bool needPrepareParams() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    [[nodiscard]] std::vector<int32_t> getAxes() const;
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

    static std::vector<float> generateTwiddlesDFT(size_t n_complex, bool inverse);
    void updateTwiddlesFFT(size_t n_complex, bool inverse);

    std::unique_ptr<jit_uni_dft_kernel> dftKernel = nullptr;
    std::unique_ptr<jit_uni_fft_kernel> fftKernel = nullptr;

    std::vector<float> twiddlesFFT;
    std::unordered_map<size_t, std::vector<float>> twiddlesMapDFT;

    std::vector<int32_t> axes;
    const size_t DATA_INDEX = 0;
    const size_t AXES_INDEX = 1;
    const size_t SIGNAL_SIZE_INDEX = 2;
    static constexpr float PI = 3.141592653589793238462643F;

    bool inverse;
    bool lastInverse;

    bool m_is_axes_size_const = false;
    bool m_is_signal_size_const = false;
};

}  // namespace ov::intel_cpu::node
