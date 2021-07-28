// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNReverseSequenceNode : public MKLDNNNode {
public:
    MKLDNNReverseSequenceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const size_t REVERSESEQUENCE_DATA = 0;
    const size_t REVERSESEQUENCE_LENGTHS = 1;

    int seq_axis;
    int batch_axis;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector srcStrides;
    size_t work_amount_dst;

    InferenceEngine::Precision lengthsPrecision;
    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
