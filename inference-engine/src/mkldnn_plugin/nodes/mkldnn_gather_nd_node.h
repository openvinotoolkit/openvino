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
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    size_t _dataRank;
    size_t _sliceRank;
    size_t _blockSize;
    size_t _batchDims;
    size_t _batchNum;
    size_t _batchStep;
    size_t _dataTypeSize;
    const size_t _dataIndex = 0;
    const size_t _indicesIndex = 1;
    std::string _errorPrefix;

    template <typename dataType>
    void gatherElementwise();
    void gatherBlocks();
};

}  // namespace MKLDNNPlugin
