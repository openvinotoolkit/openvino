// Copyright (C) 2018-2022 Intel Corporation
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

class MKLDNNEmbeddingBagPackedSumNode : public MKLDNNNode, public MKLDNNEmbeddingBagSumNode {
public:
    MKLDNNEmbeddingBagPackedSumNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    void initFromInputs() override;
    void getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) override;

    const int* _indices = nullptr;
    size_t _batch = 0;
    size_t _indicesPerBag = 0;
};

}  // namespace MKLDNNPlugin
