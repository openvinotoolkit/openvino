// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/permute_kernel.h"
#include "executors/transpose_list.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Transpose : public Node {
public:
    Transpose(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    const InferenceEngine::SizeVector& getOrder() const {
        return order;
    }

    bool isExecutable() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

    void setOptimized(bool isOptimized) {
        this->isOptimized = isOptimized;
    }

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    std::shared_ptr<ExecutorContext> transpose_context;

private:
    TransposeExecutorPtr execPtr = nullptr;
    dnnl::primitive prim;
    InferenceEngine::SizeVector order;
    ov::element::Type prec;

    TransposeParams transposeParams;

    bool isInputOrderConst = false;

    static constexpr size_t INPUT_DATA_IDX = 0lu;
    static constexpr size_t INPUT_ORDER_IDX = 1lu;

    bool performAsReorder = false;
    bool isOptimized = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
