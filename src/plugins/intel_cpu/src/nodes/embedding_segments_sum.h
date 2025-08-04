// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "embedding_bag.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class EmbeddingSegmentsSum : public Node, public EmbeddingBag {
public:
    EmbeddingSegmentsSum(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void prepareParams() override;
    [[nodiscard]] bool needShapeInfer() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    void initFromInputs() override;
    void getIndices(size_t embIndex, const int*& indices, size_t& size, int& weightsIdx, bool& withWeight) override;
    [[nodiscard]] int32_t getNumSegments() const;

    static constexpr size_t SEGMENT_ID_IDX = 2LU;
    static constexpr size_t NUM_SEGMENTS_IDX = 3LU;

    int32_t lastNumSegments_ = 0;

    const int* indices_ = nullptr;
    const int* segmentIds_ = nullptr;
    const int* defaultIndices_ = nullptr;

    size_t indicesSize_ = 0;
};

}  // namespace ov::intel_cpu::node
