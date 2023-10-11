// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <ie_precision.hpp>
#include <graph_context.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Concat : public Node {
public:
    Concat(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void selectOptimalPrimitiveDescriptor() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override { execute(strm); }
    void resolveInPlaceEdges(Edge::LOOK look) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

    bool isExecutable() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    size_t axis = 0;
    size_t reorderedAxis = 0;
    bool canBeInPlace = false;
    bool canOptimizeNspc = false;
    void execRef();
    size_t inverseOrder(const InferenceEngine::SizeVector& order, size_t axis);
    void execNspcSpecCase();
    std::vector<VectorDims> inputStrides;
    std::vector<size_t> nelemToCopy; // byte moved in each iter
    std::vector<size_t> dstOffset; // dst offset for each input
    std::vector<const uint8_t*> srcPtrs;
    bool hasOuterLoop = false;
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;
    bool canExecRef = false;
    static constexpr size_t MAX_RANK_REF = 6;
    dnnl::primitive prim;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
