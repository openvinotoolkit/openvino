// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>

namespace MKLDNNPlugin {

class MKLDNNInputNode : public MKLDNNNode {
public:
    MKLDNNInputNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNInputNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    void withMeanImage();
    MKLDNNMemoryPtr getConstBlob() const;

private:
    void cloneIfRequired(const InferenceEngine::Blob::Ptr & blob, const InferenceEngine::TensorDesc & outTensorDesc);

private:
    InferenceEngine::Precision precision;
    InferenceEngine::Blob::Ptr ieConstBlob;
    MKLDNNMemoryPtr constBlob;
    bool isMeanImage = false;
};

}  // namespace MKLDNNPlugin
