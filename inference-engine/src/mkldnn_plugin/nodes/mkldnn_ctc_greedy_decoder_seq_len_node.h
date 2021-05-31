// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNCTCGreedyDecoderSeqLenNode : public MKLDNNNode {
public:
    MKLDNNCTCGreedyDecoderSeqLenNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const size_t DATA_INDEX = 0lu;
    const size_t SEQUENCE_LENGTH_INDEX = 1lu;
    const size_t BLANK_INDEX = 2lu;
    const size_t DECODED_CLASSES_INDEX = 0lu;
    const size_t DECODED_CLASSES_LENGTH_INDEX = 1lu;
    bool mergeRepeated;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
