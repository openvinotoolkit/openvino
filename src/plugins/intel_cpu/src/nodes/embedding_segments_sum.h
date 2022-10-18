// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include "embedding_bag_sum.h"
#include <string>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class EmbeddingSegmentsSum : public Node, public EmbeddingBagSum {
public:
    EmbeddingSegmentsSum(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void prepareParams() override;
    std::vector<VectorDims> shapeInfer() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    void initFromInputs() override;
    void getIndices(int embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) override;

    const size_t SEGMENT_ID_IDX = 2lu;
    const size_t NUM_SEGMENTS_IDX = 3lu;

    int numSegments_ = 0;

    const int* indices_ = nullptr;
    const int* segmentIds_ = nullptr;
    const int* defaultIndices_ = nullptr;

    size_t indicesSize_ = 0;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
