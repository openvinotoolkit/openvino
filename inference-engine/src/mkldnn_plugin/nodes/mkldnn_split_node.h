// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNSplitNode : public MKLDNNNode {
public:
    MKLDNNSplitNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool isOptimized();
    void initOptimalPrimitiveDescriptor() override;

    void setDynamicBatchLim(int lim) override;

private:
    void prepareOptimizedParams();
    void initializeDstMemPtrs();
    void optimizedNspc2Ncsp(size_t MB);

    bool canUseOptimizedNspc2Ncsp;

    size_t axis = 1;
    std::vector<uint8_t*> dstMemPtrs;

    struct {
        std::vector<size_t> dataSize;
        std::vector<size_t> srcDataOffsets;
        size_t srcDataStride;
        size_t countStrides;
    } optimizedParams;

    size_t INPUTS_NUM = 2;
};

}  // namespace MKLDNNPlugin

