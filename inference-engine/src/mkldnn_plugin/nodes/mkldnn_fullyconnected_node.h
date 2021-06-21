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

class MKLDNNFullyConnectedNode : public MKLDNNNode {
public:
    MKLDNNFullyConnectedNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    std::vector<mkldnn::memory::format_tag> getAvailableFormatsForDims(const MKLDNNDims &dims) const override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    const std::vector<impl_desc_type>& getPrimitivesPriority() override;
    void createDescriptor(const std::vector<MKLDNNMemoryDesc>& inputDesc,
                          const std::vector<MKLDNNMemoryDesc>& outputDesc) override;

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    std::unique_ptr<MKLDNNMemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::unique_ptr<MKLDNNMemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    bool canFuse(const MKLDNNNodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    std::shared_ptr<mkldnn::primitive_attr> initPrimitiveAttr();

private:
    InferenceEngine::SizeVector weightsDims;
    InferenceEngine::SizeVector biasesDims;

    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights);

    bool withBiases = false;

    std::string errorPrefix;
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;
};

}  // namespace MKLDNNPlugin
