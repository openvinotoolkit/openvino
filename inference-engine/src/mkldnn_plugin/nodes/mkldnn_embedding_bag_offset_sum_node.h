// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include "mkldnn_embedding_bag_sum_node.h"
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNEmbeddingBagOffsetSumNode : public MKLDNNNode, public MKLDNNEmbeddingBagSumNode {
public:
    MKLDNNEmbeddingBagOffsetSumNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void initFromInputs() override;
    void getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) override;

    const size_t OFFSETS_IDX = 2lu;

    const int* indicesData_ = nullptr;
    const int* offsetsData_ = nullptr;
    const int* defaultIndices_ = nullptr;

    size_t _indicesLen;
    size_t _offsetsLen;
};

}  // namespace MKLDNNPlugin
