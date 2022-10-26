// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <ie_precision.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

class Concat : public Node {
public:
    Concat(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void selectOptimalPrimitiveDescriptor() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override { execute(strm); }

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

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
