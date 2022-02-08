// Copyright (C) 2018-2022 Intel Corporation
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
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool needPrepareParams() const override {return false;};
    bool needShapeInfer() const override {return false;};
    std::vector<VectorDims> shapeInfer() const override;
    void executeDynamicImpl(mkldnn::stream strm) override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename data_t>
    InferenceEngine::StatusCode rangeKernel();
    template <typename data_t>
    size_t getWorkAmount(data_t* startPtr = nullptr, data_t* stopPtr = nullptr, data_t* stepPtr = nullptr) const;

    static const size_t RANGE_START = 0;
    static const size_t RANGE_LIMIT = 1;
    static const size_t RANGE_DELTA = 2;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
