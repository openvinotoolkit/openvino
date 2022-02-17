// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <ie_precision.hpp>

namespace MKLDNNPlugin {

class MKLDNNConcatNode : public MKLDNNNode {
public:
    MKLDNNConcatNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void selectOptimalPrimitiveDescriptor() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }

    bool isOptimized() const;

    InferenceEngine::Precision getRuntimePrecision() const override;

    bool isExecutable() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    size_t axis = 0;
    bool canBeInPlace = false;
    bool canOptimizeNspc = false;

    size_t inverseOrder(const InferenceEngine::SizeVector& order, size_t axis);
    void execNspcSpecCase();

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;
};

}  // namespace MKLDNNPlugin

