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

class MKLDNNSoftMaxNode : public MKLDNNNode {
public:
    MKLDNNSoftMaxNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void initOptimalPrimitiveDescriptor() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    void getSupportedDescriptors() override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    void prepareParams() override;
    void executeDynamicImpl(mkldnn::stream strm) override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    size_t axis = 0;
};

}  // namespace MKLDNNPlugin

