// Copyright (C) 2018-2020 Intel Corporation
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
#include "ie_ishape_infer_extension.hpp"

namespace MKLDNNPlugin {

class MKLDNNGenericNode : public MKLDNNNode {
public:
    MKLDNNGenericNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
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
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::ILayerImplFactory::Ptr extFactory;
    InferenceEngine::IShapeInferImpl::Ptr extShapeInference;
    IE_SUPPRESS_DEPRECATED_END
    std::vector<InferenceEngine::ILayerExecImpl::Ptr> impls;
    std::map<std::string, std::string> params;
    std::map<std::string, InferenceEngine::Blob::Ptr> blobs;
};

}  // namespace MKLDNNPlugin

