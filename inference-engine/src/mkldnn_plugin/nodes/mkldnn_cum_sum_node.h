// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNCumSumNode : public MKLDNNNode {
public:
    MKLDNNCumSumNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename dataType>
    void exec();

    template <bool reverse, bool exclusive, typename dataType>
    void cumSum(const dataType *input, dataType *output, const std::vector<size_t> &strides);

    void parallelItInit(size_t start, std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    inline void parallelItStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    inline size_t getStartOffset(const std::vector<size_t> &forStartOffset, const std::vector<size_t>& strides) const;

    size_t getAxis(const InferenceEngine::Blob::CPtr& _axis, const InferenceEngine::Blob::CPtr& _data) const;

    enum { CUM_SUM_DATA, AXIS, numOfInputs };
    bool exclusive;
    bool reverse;
    size_t numOfDims;
    size_t axis = 0;
    std::vector<size_t> shape;

    InferenceEngine::Precision dataPrecision;
    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
