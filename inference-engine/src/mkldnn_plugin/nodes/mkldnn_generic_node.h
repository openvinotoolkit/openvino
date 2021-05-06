// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace MKLDNNPlugin {

class MKLDNNGenericNode : public MKLDNNNode {
public:
    MKLDNNGenericNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNGenericNode() = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool created(const MKLDNNExtensionManager::Ptr& extMgr) override;
    bool canBeInPlace() const override {
        return false;
    }

    void initDescriptor(const InferenceEngine::LayerConfig& config) override;

    void execLayer();
    void cleanup() override;

protected:
    InferenceEngine::ILayerImplFactory::Ptr extFactory;
    std::vector<InferenceEngine::ILayerExecImpl::Ptr> impls;

    const std::shared_ptr<ngraph::Node> ngraphOp;
};

}  // namespace MKLDNNPlugin

