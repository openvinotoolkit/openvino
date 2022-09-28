// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class CTCGreedyDecoderSeqLen : public Node {
public:
    CTCGreedyDecoderSeqLen(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool needPrepareParams() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const size_t DATA_INDEX = 0lu;
    const size_t SEQUENCE_LENGTH_INDEX = 1lu;
    const size_t BLANK_INDEX = 2lu;
    const size_t DECODED_CLASSES_INDEX = 0lu;
    const size_t DECODED_CLASSES_LENGTH_INDEX = 1lu;
    bool mergeRepeated;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
