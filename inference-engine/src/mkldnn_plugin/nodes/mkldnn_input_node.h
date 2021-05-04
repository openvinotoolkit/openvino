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
    MKLDNNInputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNInputNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    void withMeanImage();
    const InferenceEngine::Blob::CPtr getConstBlob() const;
    MKLDNNMemoryPtr getMemoryPtr() const;

private:
    void cloneBlobIfRequired(const MKLDNNDims& dims, const InferenceEngine::Precision& prec);

private:
    InferenceEngine::Precision precision;
    InferenceEngine::Blob::Ptr constBlob;
    MKLDNNMemoryPtr memoryPtr;
    bool isMeanImage = false;
};

}  // namespace MKLDNNPlugin
