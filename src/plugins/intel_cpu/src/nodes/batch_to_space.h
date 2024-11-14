// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class BatchToSpace : public Node {
public:
    BatchToSpace(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;

    // output shape can potentially be empty
    bool isExecutable() const override {
        return !hasEmptyInputTensors() && !hasEmptyOutputTensors();
    }

    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override { return false; };
    bool needShapeInfer() const override {return true;};
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template<typename T>
    void batchToSpaceKernel();

private:
    std::vector<size_t> blockShapeIn;
    std::vector<size_t> cropsBeginIn;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
