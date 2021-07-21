// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNMatMulNode : public MKLDNNNode {
public:
    MKLDNNMatMulNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    size_t getMaxBatch() override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    float alpha = 1.0f;
    float beta = 1.0f;
    bool transposeA = false;
    bool transposeB = false;

    int xAxis = 0;
    int yAxis = 0;

    std::vector<int> aOffsets;
    std::vector<int> bOffsets;
    std::vector<int> cOffsets;

    template<typename T0, typename T1> void process_data();

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

