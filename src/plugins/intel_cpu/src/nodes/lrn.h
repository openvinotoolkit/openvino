// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/dnnl_executor.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class Lrn : public Node {
public:
    Lrn(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    size_t descInputNumbers() override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }
    std::shared_ptr<MemoryDesc> getSrcMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    void prepareParams() override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    using executorPtr = std::shared_ptr<DnnlExecutor>;
    executorPtr execPtr = nullptr;
    dnnl::algorithm alg;
    size_t size = 1;
    int k = 1;
    float alpha = 1.0f;
    float beta = 1.0f;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
