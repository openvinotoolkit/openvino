// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNGatherTreeNode : public MKLDNNNode {
public:
    MKLDNNGatherTreeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

    template<typename DATA_T>
    void gatherTreeKernel() noexcept;

        private:
    static const size_t GATHER_TREE_STEP_IDX = 0;
    static const size_t GATHER_TREE_PARENT_IDX = 1;
    static const size_t GATHER_TREE_MAX_SEQ_LEN = 2;
    static const size_t GATHER_TREE_END_TOKEN = 3;

    InferenceEngine::Precision precision;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
