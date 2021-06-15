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

class MKLDNNSpaceToBatchNode : public MKLDNNNode {
public:
    MKLDNNSpaceToBatchNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    InferenceEngine::SizeVector inDims;
    InferenceEngine::SizeVector outDims;
    std::vector<size_t> blockShapeIn;
    std::vector<size_t> padsBeginIn;

    std::string errorPrefix;

    template<typename T>
    void SpaceToBatchKernel();
};

}  // namespace MKLDNNPlugin
