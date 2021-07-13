// Copyright (C) 2018-2021 Intel Corporation
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
    MKLDNNConvolutionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<const MemoryDesc*>& inputDesc,
                          const std::vector<const MemoryDesc*>& outputDesc) override;
    void initDescriptor(const NodeConfig& config) override;
    void createPrimitive() override;
    void selectOptimalPrimitiveDescriptor() override;
    void initSupportedPrimitiveDescriptors() override;
    void filterSupportedPrimitiveDescriptors() override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    InferenceEngine::Precision getRuntimePrecision() const override;
    std::unique_ptr<MKLDNNMemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    const mkldnn::memory& getWeights() const;
    const mkldnn::memory& getBias() const;

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return getOriginalInputsNumber();
    }

    bool canBeExecutedInInt8() const;
    size_t getGroupNum() const { return groupNum; }

    std::vector<uint8_t> inputZeroPoints;
    std::vector<float> weightsZeroPoints;
    std::vector<int32_t> outputCompensation;

    const InferenceEngine::SizeVector &getWeightDims() { return weightDims; }
    const std::vector<ptrdiff_t> &getStride() { return stride; }
    const std::vector<ptrdiff_t> &getDilation() { return dilation; }
    const std::vector<ptrdiff_t> &getPaddingL() { return paddingL; }
    const std::vector<ptrdiff_t> &getPaddingR() { return paddingR; }

    bool canFuse(const MKLDNNNodePtr& node) const override;
    bool isDepthWise() const {
        return isGrouped && 1 == groupOC && 1 == groupIC;
    }

    bool isWinograd() const { return isWino; }

protected:
    InferenceEngine::Precision fusedEltwisePrecision(const MKLDNNNodePtr& fusingNode) const;

private:
    void addZeroPoints(mkldnn::primitive_attr& attr) const;
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights) const;
    void filterSupportedDescriptors();
    bool isPossibleToSkipInitConfig(MKLDNNDescriptor &desc) const;
    bool isNspcAvailable() const;
    InferenceEngine::Blob::Ptr createInternalBlob(InferenceEngine::SizeVector dims, size_t edgeNum, bool isGrouped = false);

    bool withBiases;
    bool withSum;
    bool withDWConv;
    bool isGrouped;
    bool isPrimitivesPriorityDefined = false;
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

    size_t groupNum;
    size_t IC;
    size_t groupIC;
    size_t groupOC;

    InferenceEngine::Precision eltwisePrecision;

    const size_t X_AXIS = 0;
    const size_t Y_AXIS = 1;

    bool isWino = false;
};

}  // namespace MKLDNNPlugin

