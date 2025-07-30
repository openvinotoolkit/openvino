// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "kernels/x64/grid_sample.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class GridSample : public Node {
public:
    GridSample(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    struct threadExecParams {
        uint64_t batchNum = 1LU;
        uint64_t channelsNum = 1LU;
        std::vector<float> srcHeightF{1.F};
        std::vector<float> srcWidthF{1.F};
        std::vector<int> srcWidthB{1LU};
        std::vector<int> dataTypeSize{1LU};
        std::vector<float> srcHeightMul2F{1.F};
        std::vector<float> srcWidthMul2F{1.F};
        std::vector<float> srcHeightMul2Sub1F{1.F};
        std::vector<float> srcWidthMul2Sub1F{1.F};
        std::vector<float> srcHeightSub1F{1.F};
        std::vector<float> srcWidthSub1F{1.F};
        std::vector<float> wDenormCoefF{1.F};
        std::vector<float> hDenormCoefF{1.F};
        uint64_t gridStartB = 0LU;
        uint64_t dstStartB = 0LU;
        uint64_t srcChannelStepB = 0LU;
        uint64_t dstChannelStepB = 0LU;
        uint64_t srcBatchStepB = 0LU;
        uint64_t gridBatchStepB = 0LU;
        uint64_t dstBatchStepB = 0LU;
        uint64_t workAmount = 0LU;
        std::vector<int> buffer;
    };

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

private:
    bool alignCorners = false;
    GridSampleInterpolationMode interpolationMode = GridSampleInterpolationMode::BILINEAR;
    GridSamplePaddingMode paddingMode = GridSamplePaddingMode::ZEROS;

    uint64_t dataTypeSize = 1LU;
    uint64_t gridTypeSize = 1LU;
    ov::element::Type dataPrecision;
    ov::element::Type gridPrecision = ov::element::f32;

    size_t m_threads_num = 0LU;
    std::vector<threadExecParams> execParamsPerThread;

    static constexpr size_t IN_DATA = 0;
    static constexpr size_t IN_GRID = 1;

    std::shared_ptr<kernel::GridSampleKernelBase> jitKernel;
};

}  // namespace ov::intel_cpu::node
