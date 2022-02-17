// Copyright (C) 2018-2022 Intel Corporation
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
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    void executeDynamicImpl(mkldnn::stream strm) override;
    void prepareParams() override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    enum { CONDITION, THEN, ELSE, numOfInputs };
    enum { N, C, D, H, W, numOfDims };
    enum class SelectBroadcastType {
        NONE,
        NUMPY
    };

    SelectBroadcastType broadcastType;
    VectorDims resDims;
    VectorDims resOffset;
    VectorDims condOffset;
    VectorDims thenOffset;
    VectorDims elseOffset;

    VectorDims condDims;
    VectorDims thenDims;
    VectorDims elseDims;

    std::string errorPrefix;

    void calcOutOffset(VectorDims& offset, const VectorDims& dims);
    void calcInOffset(VectorDims& offset, const VectorDims& inDims, const VectorDims& outDims);
    template <typename COND_T, typename DATA_T>
    void execute_impl();
};

}  // namespace MKLDNNPlugin
