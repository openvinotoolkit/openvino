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

class MKLDNNDeconvolutionNode : public MKLDNNNode {
public:
    MKLDNNDeconvolutionNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<const MemoryDesc*>& inputDesc,
                          const std::vector<const MemoryDesc*>& outputDesc) override;
    void createPrimitive() override;
    void filterSupportedPrimitiveDescriptors() override;
    void filterSupportedDescriptors();
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(getParentEdges().size());
    }

    std::unique_ptr<MKLDNNMemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    std::unique_ptr<MKLDNNMemoryDesc> getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool canFuse(const MKLDNNNodePtr& node) const override;

private:
    bool withGroups = false;
    bool isDW = false;
    bool isInt8 = false;
    size_t groupNum = 1;
    size_t outDepth;
    size_t IC;
    size_t OC;
    std::vector<ptrdiff_t> kernel;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    InferenceEngine::SizeVector weightDims;
    std::vector<std::shared_ptr<mkldnn::convolution_forward::desc>> descs_fwd;
    std::vector<std::shared_ptr<mkldnn::convolution_backward_data::desc>> descs_bwd;

    mkldnn::primitive_attr attr;
    void setPostOps(mkldnn::primitive_attr &attr);

    std::string errorPrefix;

    bool canBeExecutedInInt8() const;
    InferenceEngine::Blob::Ptr createWeiBlobAsIO(InferenceEngine::SizeVector dims);
};

}  // namespace MKLDNNPlugin

