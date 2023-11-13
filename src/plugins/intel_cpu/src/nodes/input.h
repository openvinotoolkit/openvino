// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <ngraph/op/constant.hpp>
#include <string>

namespace ov {
namespace intel_cpu {
namespace node {

class Input : public Node {
public:
    Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    Input(const Shape& shape,
          const InferenceEngine::Precision& prc,
          const std::string& name,
          const std::string& type,
          const GraphContext::CPtr context);
    Input(MemoryDescPtr memDesc, const std::string& name, const std::string& type, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;

    void withMeanImage();
    MemoryCPtr getMemoryPtr() const;

    void execute(dnnl::stream strm) override {}
    void executeDynamicImpl(dnnl::stream strm) override {}
    bool isExecutable() const override {
        return false;
    }

    bool needShapeInfer() const override { return false; }
    bool needPrepareParams() const override { return false; }

private:
    void cloneBlobIfRequired();
    void initSupportedPdDefault();
    void initSupportedPdFromMemDesc();

private:
    std::shared_ptr<ngraph::op::Constant> constOp;
    MemoryCPtr memoryPtr;
    MemoryDescPtr extMemDesc = nullptr;
    bool isMeanImage = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
