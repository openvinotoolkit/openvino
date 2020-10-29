// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNFullyConnectedNode : public MKLDNNNode {
public:
    MKLDNNFullyConnectedNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNFullyConnectedNode() override = default;

    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    const std::vector<impl_desc_type>& getPrimitivesPriority() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(baseInputsNumber);
    }

    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    const mkldnn::memory& getWeights() const;
    const mkldnn::memory& getBias() const;

protected:
    std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr();

private:
    InferenceEngine::SizeVector weightsDims;
    InferenceEngine::SizeVector biasesDims;
    mkldnn::memory::format weightsFormatForSrcFormat(mkldnn::memory::format sourceFormat);

    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights);

    InferenceEngine::Blob::Ptr wScale, oScale;

    bool withBiases;
    int baseInputsNumber;
};

}  // namespace MKLDNNPlugin

