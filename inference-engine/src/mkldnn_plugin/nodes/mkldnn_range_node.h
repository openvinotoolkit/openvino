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
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override {return false;};
    bool needShapeInfer() const override {return true;};
    void prepareParams() override {};
    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }


    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    template <typename data_t>
    InferenceEngine::StatusCode rangeKernel() noexcept;
    template <typename data_t>
    size_t getWorkAmount(data_t *startPtr = nullptr, data_t *stopPtr = nullptr, data_t *stepPtr = nullptr) const noexcept;

private:
    static const size_t RANGE_START = 0;
    static const size_t RANGE_LIMIT = 1;
    static const size_t RANGE_DELTA = 2;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
