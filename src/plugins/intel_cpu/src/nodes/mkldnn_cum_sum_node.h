// Copyright (C) 2018-2022 Intel Corporation
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
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override;
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename dataType>
    void exec();

    template <bool reverse, bool exclusive, typename dataType>
    void cumSum(const dataType *input, dataType *output, const std::vector<size_t> &strides);

    void parallelItInit(size_t start, std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    inline void parallelItStep(std::vector<size_t>& counters, const std::vector<size_t>& iterationRange);

    inline size_t getStartOffset(const std::vector<size_t> &forStartOffset, const std::vector<size_t>& strides) const;

    size_t getAxis(const MKLDNNMemory& _axis, const MKLDNNMemory& _data) const;

    enum { CUM_SUM_DATA, AXIS, numOfInputs };
    bool exclusive;
    bool reverse;
    size_t numOfDims;
    size_t axis = 0;

    InferenceEngine::Precision dataPrecision;
    std::string errorPrefix;

    template<typename T>
    struct CumSumExecute {
        void operator()(MKLDNNCumSumNode* node) {
            node->exec<T>();
        }
    };
};

}  // namespace MKLDNNPlugin
