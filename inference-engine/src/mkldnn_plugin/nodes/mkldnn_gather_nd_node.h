// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGatherNDNode : public MKLDNNNode {
public:
    MKLDNNGatherNDNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    size_t batchDims = 0;
    size_t batchSize = 1;
    size_t cycles = 1;
    size_t sliceRank = 0;
    size_t dataLength = 1;

    std::vector<size_t> srcShifts;
    size_t srcBatchStride = 1;
    size_t idxBatchStride = 1;
    size_t dstBatchStride = 1;

    size_t dataTypeSize = 1;

    static constexpr size_t GATHERND_DATA = 0;
    static constexpr size_t GATHERND_INDEXES = 1;
};

}  // namespace MKLDNNPlugin
