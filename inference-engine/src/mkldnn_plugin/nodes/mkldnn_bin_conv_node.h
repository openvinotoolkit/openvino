// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNBinaryConvolutionNode : public MKLDNNNode {
public:
    MKLDNNBinaryConvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNBinaryConvolutionNode() override = default;

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
    void pushBinarizationThreshold(float value);

private:
    static Register<MKLDNNBinaryConvolutionNode> reg;
    bool withSum;
    bool withBinarization;
    bool isDW;
    bool isMerged;
    bool isGrouped;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    InferenceEngine::SizeVector weightDims;
    InferenceEngine::SizeVector biasesDims;

    ptrdiff_t dw_conv_oc;
    ptrdiff_t dw_conv_ih;
    ptrdiff_t dw_conv_iw;
    std::vector<ptrdiff_t> dw_conv_kernel;
    std::vector<ptrdiff_t> dw_conv_strides;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    float pad_value;

    std::vector<float> binarizationThresholds;
};

}  // namespace MKLDNNPlugin

