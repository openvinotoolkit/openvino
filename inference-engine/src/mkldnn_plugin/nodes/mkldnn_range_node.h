// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNRangeNode : public MKLDNNNode {
public:
    MKLDNNRangeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

    template <typename data_t>
    InferenceEngine::StatusCode rangeKernel() noexcept;
private:
    static const size_t RANGE_START = 0;
    static const size_t RANGE_LIMIT = 1;
    static const size_t RANGE_DELTA = 2;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
