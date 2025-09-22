// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "edge.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Concat : public Node {
public:
    Concat(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void selectOptimalPrimitiveDescriptor() override;
    [[nodiscard]] bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void resolveInPlaceEdges(Edge::LOOK look) override;

    [[nodiscard]] ov::element::Type getRuntimePrecision() const override;

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;
    [[nodiscard]] bool needPrepareParams() const override;
    void prepareParams() override;
    // TODO: Move to base Node class when more nodes support fuse convert
    bool supportConvertFusion() const {
        return supportFuseConvert;
    }

private:
    size_t axis = 0;
    size_t reorderedAxis = 0;
    bool canBeInPlace = false;
    bool canOptimizeNspc = false;
    bool canOptimize1DCase = false;
    void execRef();
    void execWithFuseConvert();  // Handle FP16 to FP32 conversion
    static size_t inverseOrder(const VectorDims& order, size_t axis);
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
    bool supportFuseConvert = true;  // support FP16 to FP32 conversion
    bool doFuseConvert = false;      // whether to perform FP16 to FP32 conversion
    static constexpr size_t MAX_RANK_REF = 6;
    dnnl::primitive prim;
};

}  // namespace ov::intel_cpu::node
