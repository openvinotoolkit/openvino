// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNConvolutionNode : public MKLDNNNode {
public:
    MKLDNNConvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNConvolutionNode() override = default;

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void initDescriptor(const InferenceEngine::LayerConfig& config) override;
    void createPrimitive() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights);

protected:
    void addScaleToPrimitiveAttr(mkldnn::primitive_attr attr) const;

private:
    static Register<MKLDNNConvolutionNode> reg;
    bool withBiases;
    bool withActivation;
    bool withSum;
    bool isDW;
    bool isMerged;
    bool isGrouped;
    std::vector<int> stride;
    std::vector<int> dilation;
    std::vector<int> paddingL;
    std::vector<int> paddingR;
    InferenceEngine::SizeVector weightDims;
    InferenceEngine::SizeVector biasesDims;

    int dw_conv_oc;
    int dw_conv_ih;
    int dw_conv_iw;
    std::vector<int> dw_conv_kernel;
    std::vector<int> dw_conv_strides;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    InferenceEngine::ConvolutionLayer* convLayer;
    InferenceEngine::Blob::Ptr wScale, oScale;

    bool lastInInt8Chain;
};

}  // namespace MKLDNNPlugin

