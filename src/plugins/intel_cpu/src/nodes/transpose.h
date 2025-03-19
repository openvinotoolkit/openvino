// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/permute_kernel.h"
#include "executors/transpose_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class Transpose : public Node {
public:
    Transpose(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    const VectorDims& getOrder() const {
        return order;
    }

    bool neverExecute() const override;
    bool isExecutable() const override;
    bool needPrepareParams() const override;
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

    static constexpr size_t INPUT_DATA_IDX = 0lu;
    static constexpr size_t INPUT_ORDER_IDX = 1lu;

    bool performAsReorder = false;
    bool isOptimized = false;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
