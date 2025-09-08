// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class PSROIPooling : public Node {
public:
    PSROIPooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    size_t outputDim = 0;
    size_t groupSize = 0;
    float spatialScale = 0;
    size_t pooledHeight = 0;
    size_t pooledWidth = 0;
    size_t spatialBinsX = 0;
    size_t spatialBinsY = 0;
    std::string mode;

    int channels = 0;
    int height = 0;
    int width = 0;

    int nn = 0;
    int nc = 0;
    int nh = 0;
    int nw = 0;

    // for Deformable PSROIPolling
    bool noTrans;
    int partSize = 1;
    float transStd = 1.F;

    void unpackParams(const BlockedMemoryDesc& srcDesc,
                      const BlockedMemoryDesc& dstDesc,
                      int& hInputStride,
                      int& wInputStride,
                      int& hOutputStride,
                      int& wOutputStride,
                      int& inBlockSize,
                      int& outBlockSize,
                      int& outBlockCount,
                      uint64_t& inputChannelsPadding,
                      uint64_t& outputChannelsPadding);

    template <typename inputType, typename outputType>
    void executeAverage(const inputType* srcData,
                        outputType* dstData,
                        const float* bottomRois,
                        int n,
                        int roiBatchInd,
                        const BlockedMemoryDesc& srcDesc,
                        const BlockedMemoryDesc& dstDesc);

    template <typename inputType, typename outputType>
    void executeBilinear(const inputType* srcData,
                         outputType* dstData,
                         const float* bottomRois,
                         int currentRoi,
                         int roiBatchInd,
                         const BlockedMemoryDesc& srcDesc,
                         const BlockedMemoryDesc& dstDesc);

    template <typename inputType, typename outputType>
    void executeBilinearDeformable(const inputType* srcData,
                                   outputType* dstData,
                                   const float* bottomRois,
                                   const float* bottomTrans,
                                   int numClasses,
                                   int channelsEachClass,
                                   int currentRoi,
                                   int roiBatchInd);

    template <typename inputType, typename outputType>
    void executeSpecified();

    template <typename T>
    struct PSROIPoolingExecute;
};

}  // namespace ov::intel_cpu::node
