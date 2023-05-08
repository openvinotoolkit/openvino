// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <onednn/dnnl.h>
#include <cassert>

#include <cpu/ref_eltwise.hpp>
#include <cpu/ref_depthwise_injector.hpp>
#include "utils/bfloat16.hpp"
#include "utils/cpu_utils.hpp"
#include "ie_parallel.hpp"
#include "executors/normalize_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class NormalizeL2 : public Node {
public:
    NormalizeL2(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }
    bool canFuse(const NodePtr& node) const override;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

    bool isExecutable() const override;

private:
    NormalizeL2Attrs attrs;

    dnnl::primitive_attr kernel_attrs;

    std::vector<const void*> postOpsDataPtrs;

    void setPostOps(dnnl::primitive_attr& kernel_attrs, const VectorDims& dims, bool initWeights = false);

    static constexpr size_t DATA = 0;
    static constexpr size_t AXES = 1;

    NormalizeL2ExecutorPtr execPtr = nullptr;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
