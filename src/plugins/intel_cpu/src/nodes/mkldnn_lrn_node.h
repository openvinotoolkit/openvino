// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNLrnNode : public MKLDNNNode {
public:
    MKLDNNLrnNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    size_t descInputNumbers(MKLDNNDescriptor desc) override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }
    std::shared_ptr<MemoryDesc> getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;
    std::vector<VectorDims> shapeInfer() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    mkldnn::algorithm alg;
    size_t size = 1;
    int k = 1;
    float alpha = 1.0f;
    float beta = 1.0f;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

