// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ie_common.h>
#include <node.h>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace ov {
namespace intel_cpu {
namespace node {

class Generic : public Node {
public:
    Generic(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    ~Generic() = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool created(const ExtensionManager::Ptr& extMgr) override;
    bool canBeInPlace() const override {
        return false;
    }

    void initDescriptor(const NodeConfig& config) override;

    void execLayer();
    void cleanup() override;

protected:
    NodeConfig convertLayerToNodeConfig(const InferenceEngine::LayerConfig &layerConfig);
    InferenceEngine::LayerConfig convertNodeToLayerConfig(const NodeConfig &nodeConfig);

    std::vector<InferenceEngine::ILayerExecImpl::Ptr> impls;

    const std::shared_ptr<ngraph::Node> ngraphOp;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
