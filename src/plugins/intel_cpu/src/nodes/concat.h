// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_context.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Concat : public Node {
public:
    Concat(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void selectOptimalPrimitiveDescriptor() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void resolveInPlaceEdges(Edge::LOOK look) override;

    ov::element::Type getRuntimePrecision() const override;

    bool neverExecute() const override;
    bool isExecutable() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    size_t axis = 0;
    size_t reorderedAxis = 0;
    bool canBeInPlace = false;
    bool canOptimizeNspc = false;
    bool canOptimize1DCase = false;
    void execRef();
    size_t inverseOrder(const VectorDims& order, size_t axis);
    void execNspcSpecCase();
    void exec1DCase();
    std::vector<VectorDims> inputStrides;
    std::vector<size_t> nelemToCopy;  // byte moved in each iter
    size_t nelemTotal = 0;
    std::vector<size_t> dstOffset;  // dst offset for each input
    std::vector<const uint8_t*> srcPtrs;
    bool hasOuterLoop = false;
    ov::element::Type inputPrecision = ov::element::f32;
    ov::element::Type outputPrecision = ov::element::f32;
    bool canExecRef = false;
    static constexpr size_t MAX_RANK_REF = 6;
    dnnl::primitive prim;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
