// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include "kernels/gather_uni_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGatherNode : public MKLDNNNode {
public:
    MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int axis = 0;
    int batchDims = 0;

    size_t batchSize = 1;
    size_t outerSize = 1;
    size_t srcBatchStride = 1;
    size_t idxBatchStride = 1;
    size_t dstBatchStride = 1;
    size_t dataTypeSize = 1;

    static const size_t GATHER_DATA = 0;
    static const size_t GATHER_INDEXES = 1;
    static const size_t GATHER_AXIS = 2;

    std::string errorPrefix_;
    std::shared_ptr<jitGatherKernelBase> jitKernel;

    const int shufMask8bitUni_[16]  = {0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080,
                                       0x0C080400, 0x80808080, 0x80808080, 0x80808080, 0x0C080400, 0x80808080, 0x80808080, 0x80808080};
    const int permMask8bitA2_[8]    = {0, 4, 1, 5, 2, 6, 3, 7};
    const int permMask8bitA5_[16]   = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    const int shufMask16bitUni_[16] = {0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080,
                                       0x05040100, 0x0D0C0908, 0x80808080, 0x80808080, 0x05040100, 0x0D0C0908, 0x80808080, 0x80808080};
    const int permMask16bitA2_[8]   = {0, 1, 4, 5, 2, 3, 6, 7};
    const int permMask16bitA5_[16]  = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
};

}  // namespace MKLDNNPlugin
