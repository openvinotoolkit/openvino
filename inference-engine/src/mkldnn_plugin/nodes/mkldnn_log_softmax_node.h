// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNLogSoftmaxNode : public MKLDNNNode {
public:
    MKLDNNLogSoftmaxNode(const std::shared_ptr<ngraph::Node>& op,
        const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    size_t reducedAxisSize;
    size_t reducedAxisStride = 1;
    size_t axisStep = 1;
    bool isLastDim = false;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
