// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class Reference : public Node {
public:
    Reference(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context, std::string errorMessage);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override {
        return false;
    }
    bool neverExecute() const override {
        return false;
    }
    bool isExecutable() const override {
        return true;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    ov::TensorVector prepareInputs() const;
    ov::TensorVector prepareOutputs() const;

private:
    const std::shared_ptr<ov::Node> ovCoreNode;
    const std::string additionalErrorMessage;
    bool hasOutputShapeDataDependency = false;  // flag to cache the output shape data dependency check result
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
