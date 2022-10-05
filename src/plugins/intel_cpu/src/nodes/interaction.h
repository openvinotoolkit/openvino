// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Interaction : public Node {
public:
    Interaction(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    bool isExecutable() const override;
    void executeDynamicImpl(dnnl::stream strm) override;
    void prepareParams() override;

private:
    void execRef(dnnl::stream strm);
    size_t batchSize = 0;
    size_t featureSize = 0;
    size_t inputSizes = 0;
    size_t outputFeaturesLen = 0;
    size_t interactFeatureSize = 0;
    std::string errorPrefix;
    MemoryPtr inputMemPtr;
    MemoryPtr flatMemPtr;
    MemoryPtr outputMemPtr;
    std::vector<uint32_t> featureSizes;
    InferenceEngine::Precision dataPrecision;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
