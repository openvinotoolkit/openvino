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

class MKLDNNEltwiseNode;

class MKLDNNConvolutionNode : public MKLDNNNode {
public:
    MKLDNNConvolutionNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNConvolutionNode() override = default;

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<InferenceEngine::TensorDesc>& inputDesc,
                          const std::vector<InferenceEngine::TensorDesc>& outputDesc) override;
    void initDescriptor(const InferenceEngine::LayerConfig& config) override;
    void createPrimitive() override;
    void initSupportedPrimitiveDescriptors() override;
    void filterSupportedPrimitiveDescriptors() override;
    void filterSupportedDescriptors();
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights);

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(baseInputsNumber);
    }

    int getBaseIntputsNumber() {
        return baseInputsNumber;
    }

    MKLDNNMemoryDesc getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    const mkldnn::memory& getWeights() const;
    const mkldnn::memory& getBias() const;

    bool canBeExecutedInInt8();

    std::vector<uint8_t> inputZeroPoints;
    std::vector<float> weightsZeroPoints;
    std::vector<int32_t> outputCompensation;

protected:
    void addScaleToPrimitiveAttr(mkldnn::primitive_attr attr) const;
    InferenceEngine::Precision fusedEltwisePrecision(MKLDNNEltwiseNode *eltwiseNode, int findex);

private:
    mkldnn::memory::data_type precisionToDataType(InferenceEngine::Precision prec);
    void addZeroPoints(mkldnn::primitive_attr& attr) const;

    bool withBiases;
    bool withSum;
    bool withDWConv;
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
    mkldnn::memory::data_type dw_conv_in_dt;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    InferenceEngine::Blob::Ptr wScale, oScale;

    size_t groupNum;
    int baseInputsNumber;

    InferenceEngine::Precision eltwisePrecision;
};

}  // namespace MKLDNNPlugin

