// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <mkldnn_extension_utils.h>

namespace MKLDNNPlugin {

enum AdaPoolingMode { AVG, MAX };

class MKLDNNAdaPoolingNode : public MKLDNNNode {
public:
    MKLDNNAdaPoolingNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    std::vector<mkldnn::memory::format_tag> getAvailableFormatsForDims(const MKLDNNDims &dims) const override;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int spatialDimsCount;
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;
    template <typename inputType, typename outputType>
    void executeSpecified();
    template<typename T>
    struct AdaPoolingExecute;
    inline void setBinBorders(size_t *startPtr, size_t *endPtr, size_t idx, size_t inputLength, size_t outputLength);
    template <typename T>
    T avgDiv(const T sum, size_t binSize);

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
