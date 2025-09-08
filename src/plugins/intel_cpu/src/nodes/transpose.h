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
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/transpose.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class Transpose : public Node {
public:
    Transpose(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;
    [[nodiscard]] bool canBeInPlace() const override {
        return false;
    }

    [[nodiscard]] const VectorDims& getOrder() const {
        return order;
    }

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;
    [[nodiscard]] bool needPrepareParams() const override;
    void prepareParams() override;

    void setOptimized(bool isOptimized) {
        this->isOptimized = isOptimized;
    }

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    std::shared_ptr<ExecutorContext> transpose_context;

private:
    TransposeExecutorPtr execPtr = nullptr;
    dnnl::primitive prim;
    VectorDims order;
    ov::element::Type prec;

    TransposeParams transposeParams;

    bool isInputOrderConst = false;

    static constexpr size_t INPUT_DATA_IDX = 0LU;
    static constexpr size_t INPUT_ORDER_IDX = 1LU;

    bool performAsReorder = false;
    bool isOptimized = false;
};

}  // namespace ov::intel_cpu::node
