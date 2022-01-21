// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNBatchToSpaceNode : public MKLDNNNode {
public:
    MKLDNNBatchToSpaceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override { return false; };
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    template<typename T>
    void batchToSpaceKernel();

private:
    std::vector<size_t> blockShapeIn;
    std::vector<size_t> cropsBeginIn;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
