// Copyright (C) 2018-2022 Intel Corporation
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
    MKLDNNFullyConnectedNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    std::vector<mkldnn::memory::format_tag> getAvailableFormatsForDims(const Shape &dims) const override;
    void getSupportedDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    size_t getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() == 3 ? 2 : 1;
    }

    const std::vector<impl_desc_type>& getPrimitivesPriority() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    void initSupportedPrimitiveDescriptors() override;
    std::shared_ptr<MemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::shared_ptr<MemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    bool canFuse(const MKLDNNNodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr() override;

    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;

    void setDynamicBatchLim(int lim) override;

private:
    void createDescriptorInternal(const mkldnn::memory::desc &inputDesc,
                                  const mkldnn::memory::desc &outputDesc);

    VectorDims makeDummyInputDims() const;
    VectorDims makeDummyOutputDims(const VectorDims& inDims) const;

    VectorDims inDims;
    VectorDims outDims;

    void setPostOps(mkldnn::primitive_attr &attr, const VectorDims &dims, bool initWeights = false);

    bool withBiases = false;

    std::string errorPrefix;
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;
};

}  // namespace MKLDNNPlugin
