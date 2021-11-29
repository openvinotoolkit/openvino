// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNSelectNode : public MKLDNNNode {
public:
    MKLDNNSelectNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    enum { CONDITION, THEN, ELSE, numOfInputs };
    enum { N, C, D, H, W, numOfDims };
    enum class SelectBroadcastType {
        NONE,
        NUMPY
    };

    SelectBroadcastType broadcastType;
    std::vector<size_t> resDims;
    std::vector<size_t> resOffset;
    std::vector<size_t> condOffset;
    std::vector<size_t> thenOffset;
    std::vector<size_t> elseOffset;

    std::string errorPrefix;

    void calcOutOffset(std::vector<size_t>& offset, const std::vector<size_t>& dims);
    void calcInOffset(std::vector<size_t>& offset, const std::vector<size_t>& inDims, const std::vector<size_t>& outDims);
    template <typename COND_T, typename DATA_T>
    void execute_impl();
};

}  // namespace MKLDNNPlugin
