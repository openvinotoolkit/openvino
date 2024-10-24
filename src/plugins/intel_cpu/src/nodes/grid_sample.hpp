// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/x64/grid_sample.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class GridSample : public Node {
public:
    GridSample(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    struct threadExecParams {
        uint64_t batchNum    = 1lu;
        uint64_t channelsNum = 1lu;
        std::vector<float> srcHeightF{ 1.f };
        std::vector<float> srcWidthF{ 1.f };
        std::vector<int>   srcWidthB{ 1lu };
        std::vector<int>   dataTypeSize{ 1lu };
        std::vector<float> srcHeightMul2F{ 1.f };
        std::vector<float> srcWidthMul2F{ 1.f };
        std::vector<float> srcHeightMul2Sub1F{ 1.f };
        std::vector<float> srcWidthMul2Sub1F{ 1.f };
        std::vector<float> srcHeightSub1F{ 1.f };
        std::vector<float> srcWidthSub1F{ 1.f };
        std::vector<float> wDenormCoefF{ 1.f };
        std::vector<float> hDenormCoefF{ 1.f };
        uint64_t gridStartB      = 0lu;
        uint64_t dstStartB       = 0lu;
        uint64_t srcChannelStepB = 0lu;
        uint64_t dstChannelStepB = 0lu;
        uint64_t srcBatchStepB   = 0lu;
        uint64_t gridBatchStepB  = 0lu;
        uint64_t dstBatchStepB   = 0lu;
        uint64_t workAmount      = 0lu;
        std::vector<int> buffer;
    };

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    void prepareParams() override;

private:
    bool alignCorners = false;
    GridSampleInterpolationMode interpolationMode = GridSampleInterpolationMode::BILINEAR;
    GridSamplePaddingMode paddingMode = GridSamplePaddingMode::ZEROS;

    uint64_t dataTypeSize = 1lu;
    uint64_t gridTypeSize = 1lu;
    ov::element::Type dataPrecision;
    ov::element::Type gridPrecision = ov::element::f32;

    size_t m_threads_num = 0lu;
    std::vector<threadExecParams> execParamsPerThread;

    static constexpr size_t IN_DATA = 0;
    static constexpr size_t IN_GRID = 1;

    std::shared_ptr<kernel::GridSampleKernelBase> jitKernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
